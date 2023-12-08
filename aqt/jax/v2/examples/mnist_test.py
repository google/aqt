# Copyright 2022 Google LLC
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

"""Test for mnist."""

import copy
from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax.v2 import config
from aqt.jax.v2.examples import mnist
from aqt.jax.v2.flax import aqt_flax
import jax
import jax.numpy as jnp


class MnistTest(parameterized.TestCase):

  def test_mnist_training(self):
    # TODO(lew): use einsum in mnist test
    target_loss = {
        "cpu": [
            3.981118679046630859375000000000,  # rome, milan
            3.981118917465209960937500000000,  # skylake
        ],
        "TPU v2": [3.991446971893310546875000000000],
        "TPU v3": [3.991446971893310546875000000000],
        "TPU v4": [3.992439270019531250000000000000],
        "TPU v5 lite": [3.991421222686767578125000000000],
    }

    aqt_cfg = aqt_flax.config_v4(
        drhs_bits=8,
        drhs_accumulator_dtype=jnp.int32,  # overwrite the default None
    )
    # below 3 lines are differences between config_v4/v3 and fully_quantized
    config.set_stochastic_rounding(aqt_cfg, True, True, "jax.uniform")
    aqt_cfg.dlhs.rhs.use_fwd_quant = True
    aqt_cfg.drhs.rhs.use_fwd_quant = True

    def forward(model, apply_fn):
      return apply_fn(
          model,
          ds["image"],
          rngs={"params": jax.random.PRNGKey(0)},
          mutable=True,
      )

    # RNGs
    rng = jax.random.key(0)
    rng, init_rng = jax.random.split(rng)
    rng, ds_rng = jax.random.split(rng)
    rng, input_rng = jax.random.split(rng)
    del rng

    # Dataset
    ds_size = 8
    ds = {
        "image": jax.random.uniform(key=ds_rng, shape=(ds_size, 28, 28, 1)),
        "label": jax.random.randint(
            key=ds_rng, shape=(ds_size,), minval=0, maxval=10
        ),
    }

    # Stage 1: regular training
    state = mnist.create_train_state(init_rng, aqt_cfg)

    state, train_loss, _ = mnist.train_epoch(
        state, ds, batch_size=ds_size // 2, rng=input_rng
    )

    assert train_loss in target_loss[jax.devices()[0].device_kind]

    # Run forward once more in the same mode to get logits for testing below.
    logits_s1, _ = forward(state.model, state.cnn_eval.apply)

    # Stage 2: Model conversion (quantized weights freezing)

    apply_serving, model_serving = mnist.serving_conversion(state)

    expected_aqt_pytree = {
        "Dense_0": {
            "AqtDotGeneral_0": {
                "rhs": {"frozen": jnp.int8},
                "rhs_scale": {"frozen": jnp.float32},
            }
        },
        "Dense_1": {
            "AqtDotGeneral_0": {
                "rhs": {"frozen": jnp.int8},
                "rhs_scale": {"frozen": jnp.float32},
            }
        },
    }

    serving_pytree = jax.tree_util.tree_map(lambda x: x.dtype, model_serving)
    assert "aqt" in serving_pytree.keys(), serving_pytree
    assert serving_pytree["aqt"] == expected_aqt_pytree, serving_pytree

    def zero_out_params(model, layer: str):
      updated_model = copy.deepcopy(model)
      updated_model["params"][layer]["kernel"] = jnp.zeros_like(
          updated_model["params"][layer]["kernel"]
      )
      return updated_model

    # We can, but do not have to delete unquantized weights.
    model_serving = zero_out_params(model_serving, "Dense_0")
    model_serving = zero_out_params(model_serving, "Dense_1")

    # Stage 3: inference mode.
    logits_s3, _ = forward(model_serving, apply_serving)
    assert (logits_s3 == logits_s1).all()

    # Sanity check 1: We can't zero out Conv_0 because it was not frozen.
    sanity_model = zero_out_params(model_serving, "Conv_0")
    bad_logits, _ = forward(sanity_model, apply_serving)
    assert not (bad_logits == logits_s3).all()

    # Sanity check 2: Frozen weights are indeed used for inference.
    #   If we zero them out, loss would change.
    model_serving["aqt"]["Dense_0"]["AqtDotGeneral_0"]["rhs"]["frozen"] = (
        jnp.zeros_like(
            model_serving["aqt"]["Dense_0"]["AqtDotGeneral_0"]["rhs"]["frozen"]
        )
    )
    bad_logits, _ = forward(model_serving, apply_serving)
    assert not (bad_logits == logits_s3).all()


if __name__ == "__main__":
  absltest.main()
