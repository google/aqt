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

"""Test for flax e2e model."""
import copy
from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax.v2 import config
from aqt.jax.v2.examples import flax_e2e_model
from aqt.jax.v2.flax import aqt_flax
import jax
import jax.numpy as jnp


class MnistTest(parameterized.TestCase):

  def test_mnist_training(self):
    target_loss = {
        "cpu": [
            3.931983232498168945312500000000,
            3.932066917419433593750000000000,
        ],
        "TPU v2": [3.950709819793701171875000000000],
        "TPU v3": [3.950709819793701171875000000000],
        "TPU v4": [3.950191974639892578125000000000],
        "TPU v5 lite": [3.949246168136596679687500000000],
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
    state = flax_e2e_model.create_train_state(init_rng, aqt_cfg)

    state, train_loss, _ = flax_e2e_model.train_epoch(
        state, ds, batch_size=ds_size // 2, rng=input_rng
    )

    device_kind = jax.devices()[0].device_kind
    expected_train_loss = target_loss[device_kind]
    if train_loss not in expected_train_loss:
      msg = "train_loss changed. Consider updating with the following:\n"
      msg += f'        "{device_kind}": [{train_loss:.30f}]'
      self.fail(msg)

    # Run forward once more in the same mode to get logits for testing below.
    logits_s1, _ = forward(state.model, state.cnn_eval.apply)

    # Stage 2: Model conversion (quantized weights freezing)

    apply_serving, model_serving = flax_e2e_model.serving_conversion(state)

    dtype = jnp.dtype
    expected_aqt_pytree = {
        "aqt": {
            "AqtEinsum_0": {
                "AqtDotGeneral_0": {
                    "qlhs": {
                        "scale": (dtype("float32"), (1, 10)),
                        "value": (dtype("int8"), (10, 10)),
                    }
                }
            },
            "Dense_0": {
                "AqtDotGeneral_0": {
                    "qrhs": {
                        "scale": (dtype("float32"), (1, 256)),
                        "value": (dtype("int8"), (3136, 256)),
                    }
                }
            },
            "Dense_1": {
                "AqtDotGeneral_0": {
                    "qrhs": {
                        "scale": (dtype("float32"), (1, 10)),
                        "value": (dtype("int8"), (256, 10)),
                    }
                }
            },
        },
        "batch_stats": {
            "BatchNorm_0": {
                "mean": (dtype("float32"), (32,)),
                "var": (dtype("float32"), (32,)),
            },
            "BatchNorm_1": {
                "mean": (dtype("float32"), (64,)),
                "var": (dtype("float32"), (64,)),
            },
        },
        "params": {
            "BatchNorm_0": {
                "bias": (dtype("float32"), (32,)),
                "scale": (dtype("float32"), (32,)),
            },
            "BatchNorm_1": {
                "bias": (dtype("float32"), (64,)),
                "scale": (dtype("float32"), (64,)),
            },
            "Conv_0": {
                "bias": (dtype("float32"), (32,)),
                "kernel": (dtype("float32"), (3, 3, 1, 32)),
            },
            "Conv_1": {
                "bias": (dtype("float32"), (64,)),
                "kernel": (dtype("float32"), (3, 3, 32, 64)),
            },
            "Dense_0": {
                "bias": (dtype("float32"), (256,)),
                "kernel": (dtype("float32"), (3136, 256)),
            },
            "Dense_1": {
                "bias": (dtype("float32"), (10,)),
                "kernel": (dtype("float32"), (256, 10)),
            },
        },
    }

    serving_pytree = jax.tree_util.tree_map(
        lambda x: (x.dtype, x.shape), model_serving
    )
    if serving_pytree != expected_aqt_pytree:
      print()
      print("serving_pytree:      ", serving_pytree)
      print("expected_aqt_pytree: ", expected_aqt_pytree)
      assert False, "serving_pytree != expected_aqt_pytree"

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
    model_serving["aqt"]["Dense_0"]["AqtDotGeneral_0"]["qrhs"]["value"] = (
        jnp.zeros_like(
            model_serving["aqt"]["Dense_0"]["AqtDotGeneral_0"]["qrhs"]["value"]
        )
    )
    bad_logits, _ = forward(model_serving, apply_serving)
    assert not (bad_logits == logits_s3).all()


if __name__ == "__main__":
  absltest.main()
