# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test for AQT state updator."""
from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax.v2 import config
from aqt.jax.v2.flax import aqt_state_updator
import flax.linen as nn
import jax
import jax.numpy as jnp


def test_eq(name, a, b):
  mean_err = jnp.mean(jnp.abs(a - b))
  if mean_err > 1e-6:
    print("mean_err =", mean_err)
    print(f"FAIL: {name}")
    assert False


class DotGeneralStaticRangeStateUpdatorTest(parameterized.TestCase):

  def test_update(self):
    dimension_numbers = (((2,), (0,)), ((), ()))
    moving_average_weight = 0.9
    lhs_shape = (3, 4, 5)
    rhs_shape = (5, 6)

    class Model(nn.Module):
      aqt_cfg: config.DotGeneral | None

      @nn.compact
      def __call__(self, lhs, rhs):
        lhs_updator = aqt_state_updator.DotGeneralStaticRangeStateUpdator(
            cfg=cfg.fwd,
            is_lhs=True,
            lhs_shape=lhs.shape,
            rhs_shape=rhs.shape,
            lhs_dtype=lhs.dtype,
            rhs_dtype=rhs.dtype,
            dimension_numbers=dimension_numbers,
            quant_collection="aqt",
            moving_average_weight=moving_average_weight
        )
        rhs_updator = aqt_state_updator.DotGeneralStaticRangeStateUpdator(
            cfg=cfg.fwd,
            is_lhs=False,
            lhs_shape=lhs.shape,
            rhs_shape=rhs.shape,
            lhs_dtype=lhs.dtype,
            rhs_dtype=rhs.dtype,
            dimension_numbers=dimension_numbers,
            quant_collection="aqt",
            moving_average_weight=moving_average_weight
        )

        lhs_updator.update(lhs, rhs)
        rhs_updator.update(lhs, rhs)
        return lhs_updator.get_state().max, rhs_updator.get_state().max

    cfg = config.config_v4()
    cfg.fwd.lhs.quantizer.calib_shared_axes = "per_tensor"

    key = jax.random.PRNGKey(0)
    key, sub = jax.random.split(key, 2)
    model = Model(aqt_cfg=cfg)
    model_params = model.init(sub, jnp.zeros(lhs_shape), jnp.zeros(rhs_shape))

    @jax.jit
    def update(params, lhs, rhs):
      return model.apply(params, lhs, rhs, mutable=True)

    def apply(key, params):
      key, sub1, sub2 = jax.random.split(key, 3)
      lhs = jax.random.normal(sub1, lhs_shape)
      rhs = jax.random.normal(sub2, rhs_shape)
      lhs_max = jnp.max(jnp.abs(lhs), keepdims=True)
      rhs_max = jnp.max(jnp.abs(rhs), axis=[0], keepdims=True)

      (lhs_stat_max, rhs_stat_max), params = update(params, lhs, rhs)

      return params, key, lhs_max, rhs_max, lhs_stat_max, rhs_stat_max

    # 1. After initial update, the retrieved max values should be the same.
    model_params, key, lhs_max, rhs_max, lhs_stat_max, rhs_stat_max = apply(
        key, model_params
    )
    test_eq("lhs", lhs_max, lhs_stat_max)
    test_eq("rhs", rhs_max, rhs_stat_max)

    # 2. After second update, the retrieved  max values should be updated with
    #    the moving average.
    _, _, lhs_max_2, rhs_max_2, lhs_stat_max, rhs_stat_max = apply(
        key, model_params
    )
    test_eq(
        "lhs",
        lhs_max * moving_average_weight
        + lhs_max_2 * (1.0 - moving_average_weight),
        lhs_stat_max,
    )
    test_eq(
        "rhs",
        rhs_max * moving_average_weight
        + rhs_max_2 * (1.0 - moving_average_weight),
        rhs_stat_max,
    )


if __name__ == "__main__":
  absltest.main()
