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
from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax.v2 import config
from aqt.jax.v2.examples import flax_e2e_model
import aqt.jax.v2.numerics.fp8_numerics as numerics
import jax
import jax.numpy as jnp
import numpy as np


class MyTest(parameterized.TestCase):

  def test_2bit_mantissa(self):
    x = jnp.array(
        [
            [0, 1, -1, 0.1234567, 100, 0.5, 0.00000000001, 0.01],
            [
                -2.9098532,
                -0.8508397,
                -0.14903745,
                -3.572757,
                0.63277215,
                0.7478278,
                1.3707844,
                1.3319879,
            ],
        ],
        dtype=jnp.float32,
    )

    result = numerics.round_to_nearest_even(x, jnp.float8_e5m2)

    np.testing.assert_array_equal(
        result,
        np.array(
            [
                [0, 1, -1, 0.125, 96, 0.5, 0, 10 / 1024],
                [-3, -0.875, -0.15625, -3.5, 0.625, 0.75, 1.25, 1.25],
            ],
            dtype=jnp.float32,
        ),
    )

  def test_3bit_mantissa(self):
    x = jnp.array(
        [
            [0, 1, -1, 0.1234567, 100, 0.5, 0.00000000001, 0.01],
            [
                -2.9098532,
                -0.8508397,
                -0.14903745,
                -3.572757,
                0.63277215,
                0.7478278,
                1.3707844,
                1.3319879,
            ],
        ],
        dtype=jnp.float32,
    )

    result = numerics.round_to_nearest_even(x, jnp.float8_e4m3fn)

    np.testing.assert_array_equal(
        result,
        np.array(
            [
                [0, 1, -1, 0.125, 96, 0.5, 0, 10 / 1024],
                [-3, -0.875, -0.15625, -3.5, 0.625, 0.75, 1.375, 1.375],
            ],
            dtype=jnp.float32,
        ),
    )

  def test_retains_dtype(self):
    result_f16 = numerics.round_to_nearest_even(
        jnp.array([1], dtype=jnp.bfloat16), jnp.float8_e5m2
    )
    result_f32 = numerics.round_to_nearest_even(
        jnp.array([1], dtype=jnp.float32), jnp.float8_e4m3fn
    )

    self.assertEqual(result_f32.dtype, jnp.float8_e4m3fn)
    self.assertEqual(result_f16.dtype, jnp.float8_e5m2)

  @parameterized.parameters([
      dict(fwd_bits="e4m3"),
      dict(fwd_bits="e5m2"),
  ])
  def test_mnist_training(self, fwd_bits: str):
    target_loss = {
        "e4m3": {
            "cpu": [
                # One of them is: milan, rome, haswell
                # Other is: skylake, cascadelake
                3.164344310760498046875000000000,
                3.164344787597656250000000000000,
            ],
            "TPU v2": [3.209973335266113281250000000000],
            "TPU v3": [3.209973335266113281250000000000],
            "TPU v4": [3.210477352142333984375000000000],
            "TPU v5 lite": [3.210477113723754882812500000000],
        },
        "e5m2": {
            "cpu": [
                # One of them is: milan, rome, haswell
                # Other is: skylake, cascadelake
                3.087945222854614257812500000000,
                3.087944984436035156250000000000,
            ],
            "TPU v2": [3.169983625411987304687500000000],
            "TPU v3": [3.169983625411987304687500000000],
            "TPU v4": [3.169979095458984375000000000000],
            "TPU v5 lite": [3.169981479644775390625000000000],
        },
    }

    # RNGs
    rng = jax.random.key(0)
    rng, init_rng = jax.random.split(rng)
    rng, image_rng = jax.random.split(rng)
    rng, label_rng = jax.random.split(rng)
    rng, input_rng = jax.random.split(rng)
    del rng

    # Dataset
    ds_size = 8
    ds = {
        "image": jax.random.uniform(key=image_rng, shape=(ds_size, 28, 28, 1)),
        "label": jax.random.randint(
            key=label_rng, shape=(ds_size,), minval=0, maxval=10
        ),
    }

    aqt_cfg = config.config_fwd_fp8(fwd_bits)
    state = flax_e2e_model.create_train_state(init_rng, aqt_cfg)

    _, train_loss, _ = flax_e2e_model.train_epoch(
        state, ds, batch_size=ds_size // 2, rng=input_rng
    )

    device = jax.devices()[0].device_kind
    if train_loss not in target_loss[fwd_bits][device]:
      msg = f"train_loss={train_loss:.30f}, {device=}, {fwd_bits=}"
      self.fail(msg)


if __name__ == "__main__":
  absltest.main()
