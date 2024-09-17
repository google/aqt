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
import functools

from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax.v2 import aqt_tensor
from aqt.jax.v2 import config
from aqt.jax.v2 import utils
from aqt.jax.v2.examples import flax_e2e_model
from aqt.jax.v2.flax import aqt_flax_calibration
from aqt.jax.v2.flax import delayed_scaling_calibration
import jax
import jax.numpy as jnp
import numpy as np


def _dummy_dataset(ds_size, image_rng, label_rng):
  return {
      "image": jax.random.uniform(key=image_rng, shape=(ds_size, 28, 28, 1)),
      "label": jax.random.randint(
          key=label_rng, shape=(ds_size,), minval=0, maxval=10
      ),
  }


class MnistTest(parameterized.TestCase):

  # Unable to use config_v4() in parameters since it needs jax.device info.
  # TODO(aqt): Move config_v4() into parameters once int4 works for cpu.
  @parameterized.parameters([
      (
          {
              "drhs_bits": 8,
              "drhs_accumulator_dtype": jnp.int32,  # overwrite the default None
          },
          8,
      ),
      (
          {
              "fwd_bits": 4,
              "fwd_accumulator_dtype": None,
              "dlhs_accumulator_dtype": None,
          },
          4,
      ),
  ])
  def test_mnist_training(self, configs, bits):
    aqt_cfg_dg = config.config_v4(**configs)
    target_loss = {
        8: {
            "cpu": [
                3.122317314147949218750000000000,
                3.122316360473632812500000000000,
                3.122316837310791015625000000000,  # colab
            ],
            "TPU v2": [3.198328018188476562500000000000],
            "TPU v3": [3.198328018188476562500000000000],
            "TPU v4": [3.198297500610351562500000000000],
            "TPU v5 lite": [3.198297500610351562500000000000],
        },
        4: {
            "cpu": [2.258865118026733398437500000000],
            "TPU v2": [2.302409172058105468750000000000],
            "TPU v3": [2.302409172058105468750000000000],
            "TPU v4": [2.302409172058105468750000000000],
            "TPU v5 lite": [2.302409172058105468750000000000],
        },
    }
    # below 3 lines are differences between config_v4/v3 and fully_quantized
    config.set_stochastic_rounding(aqt_cfg_dg, True, True, "jax.uniform")
    aqt_cfg_dg.dlhs.rhs.use_fwd_quant = True
    aqt_cfg_dg.drhs.rhs.use_fwd_quant = True

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
    rng, image_rng = jax.random.split(rng)
    rng, label_rng = jax.random.split(rng)
    rng, input_rng = jax.random.split(rng)
    del rng

    # Dataset
    ds_size = 8
    ds = _dummy_dataset(ds_size, image_rng, label_rng)

    # Stage 1: regular training
    state = flax_e2e_model.create_train_state(init_rng, aqt_cfg_dg=aqt_cfg_dg)

    state, train_loss, _ = flax_e2e_model.train_epoch(
        state, ds, batch_size=ds_size // 2, rng=input_rng
    )

    device_kind = jax.devices()[0].device_kind
    expected_train_loss = target_loss[bits][device_kind]
    if train_loss not in expected_train_loss:
      msg = "train_loss changed. Consider updating with the following:\n"
      msg += f'        "{device_kind}": [{train_loss:.30f}]'
      self.fail(msg)

    # Run forward once more in the same mode to get logits for testing below.
    logits_s1, _ = forward(state.model, state.cnn_eval.apply)

    # Stage 2: Model conversion (quantized weights freezing)

    apply_serving, model_serving = flax_e2e_model.serving_conversion(state)

    dtype = jnp.dtype
    expected_dtype = jnp.int4 if bits == 4 else jnp.int8
    expected_aqt_pytree = {
        "aqt": {
            "AqtEinsum_0": {
                "AqtDotGeneral_0": {
                    "qlhs": {
                        "frozen": aqt_tensor.QTensor(
                            qvalue=(expected_dtype, (2, 5, 10)),
                            scale=[(dtype("float32"), (2, 1, 10))],
                            scale_t=None,
                            bias=[],
                            dequant_dtype=dtype("float32"),
                        )
                    }
                }
            },
            "Dense_0": {
                "AqtDotGeneral_0": {
                    "qrhs": {
                        "frozen": aqt_tensor.QTensor(
                            # The weight shape was (3136, 256) before tiling.
                            # After tiling it is (2, 1568, 256).
                            # Contraction shape 3136 is tiled to (2, 1568).
                            # The remaining shape 256 is not tiled.
                            qvalue=(expected_dtype, (2, 1568, 256)),
                            scale=[(dtype("float32"), (2, 1, 256))],
                            # The scale_t shape was (1, 256) before tiling.
                            # After tiling the scale shape is (2, 1, 256),
                            # then transposed to (2, 1, 256).
                            scale_t=None,
                            bias=[],
                            dequant_dtype=dtype("float32"),
                        )
                    }
                }
            },
            "Dense_1": {
                "AqtDotGeneral_0": {
                    "qrhs": {
                        "frozen": aqt_tensor.QTensor(
                            # The weight shape was (256, 10) before tiling.
                            # After tiling it is (2, 128, 10).
                            # Contraction shape 256 is tiled to (2, 128).
                            # The remaining shape 10 is not tiled.
                            qvalue=(expected_dtype, (2, 128, 10)),
                            scale=[(dtype("float32"), (2, 1, 10))],
                            # The scale_t shape was (1, 10) before tiling.
                            # After tiling the scale shape is (2, 1, 10),
                            # then transposed to (2, 1, 10).
                            scale_t=None,
                            bias=[],
                            dequant_dtype=dtype("float32"),
                        )
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
    qt = model_serving["aqt"]["Dense_0"]["AqtDotGeneral_0"]["qrhs"]["frozen"]
    qt.qvalue = jnp.zeros(qt.qvalue.shape).astype(qt.qvalue.dtype)
    bad_logits, _ = forward(model_serving, apply_serving)
    assert not (bad_logits == logits_s3).all()

  @parameterized.parameters([
      (
          {
              "fwd_bits": 8,
              "fwd_accumulator_dtype": None,
              "dlhs_accumulator_dtype": None,
          },
          8,
      ),
      (
          {
              "fwd_bits": 4,
              "fwd_accumulator_dtype": None,
              "dlhs_accumulator_dtype": None,
          },
          4,
      ),
  ])
  def test_mnist_training_quantized_conv(self, configs, bits):
    aqt_cfg_dg = config.config_v4(**configs)
    aqt_cfg_conv = config.conv_general_dilated_make(
        2, lhs_bits=bits, rhs_bits=bits, initialize_calibration=False
    )
    if aqt_cfg_conv.lhs:
      aqt_cfg_conv.dg_quantizer.lhs.init_calibration()
    if aqt_cfg_conv.rhs:
      aqt_cfg_conv.dg_quantizer.rhs.init_calibration()
    device_kind = jax.devices()[0].device_kind
    if device_kind == "cpu" and bits == 4:
      # Some 4-bit operations are not supported on cpu.
      # Omitting tests on cpu with 4-bits.
      return
    target_loss = {
        8: {
            "cpu": [2.302584648132324218750000000000],
            "TPU v2": [2.302581071853637695312500000000],
            "TPU v3": [2.302581071853637695312500000000],
            "TPU v4": [2.302581071853637695312500000000,],
            "TPU v5 lite": [2.302581071853637695312500000000],
        },
        4: {
            "TPU v2": [2.302409172058105468750000000000],
            "TPU v3": [2.302409172058105468750000000000],
            "TPU v4": [2.302409172058105468750000000000],
            "TPU v5 lite": [2.302409172058105468750000000000],
        },
    }
    # below 3 lines are differences between config_v4/v3 and fully_quantized
    config.set_stochastic_rounding(aqt_cfg_dg, True, True, "jax.uniform")
    aqt_cfg_dg.dlhs.rhs.use_fwd_quant = True
    aqt_cfg_dg.drhs.rhs.use_fwd_quant = True

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
    rng, image_rng = jax.random.split(rng)
    rng, label_rng = jax.random.split(rng)
    rng, input_rng = jax.random.split(rng)
    del rng

    # Dataset
    ds_size = 8
    ds = _dummy_dataset(ds_size, image_rng, label_rng)

    # Stage 1: regular training
    state = flax_e2e_model.create_train_state(
        init_rng, aqt_cfg_dg=aqt_cfg_dg, aqt_cfg_conv=aqt_cfg_conv)

    state, train_loss, _ = flax_e2e_model.train_epoch(
        state, ds, batch_size=ds_size // 2, rng=input_rng
    )

    device_kind = jax.devices()[0].device_kind
    expected_train_loss = target_loss[bits][device_kind]
    if train_loss not in expected_train_loss:
      msg = "train_loss changed. Consider updating with the following:\n"
      msg += f'        "{device_kind}": [{train_loss:.30f}]'
      self.fail(msg)

    # Run forward once more in the same mode to get logits for testing below.
    logits_s1, _ = forward(state.model, state.cnn_eval.apply)

    # Stage 2: Model conversion (quantized weights freezing)

    apply_serving, model_serving = flax_e2e_model.serving_conversion(state)

    dtype = jnp.dtype
    expected_dtype = jnp.int4 if bits == 4 else jnp.int8
    expected_aqt_pytree = {
        "aqt": {
            "AqtEinsum_0": {
                "AqtDotGeneral_0": {
                    "qlhs": {
                        "frozen": aqt_tensor.QTensor(
                            qvalue=(expected_dtype, (2, 5, 10)),
                            scale=[(dtype("float32"), (2, 1, 10))],
                            scale_t=None,
                            bias=[],
                            dequant_dtype=dtype("float32"),
                            tiling_state=None,
                        )
                    }
                }
            },
            "Conv_0": {
                "AqtConvGeneralDilated_0": {
                    "qrhs": {
                        "frozen": aqt_tensor.QTensor(
                            qvalue=(expected_dtype, (3, 3, 1, 32)),
                            scale=[(dtype("float32"), (1, 1, 1, 32))],
                            scale_t=None,
                            bias=[],
                            dequant_dtype=dtype("float32"),
                            tiling_state=None)
                    }
                }
            },
            "Conv_1": {
                "AqtConvGeneralDilated_0": {
                    "qrhs": {"frozen": aqt_tensor.QTensor(
                        qvalue=(expected_dtype, (3, 3, 32, 64)),
                        scale=[(dtype("float32"), (1, 1, 1, 64))],
                        scale_t=None,
                        bias=[],
                        dequant_dtype=dtype("float32"),
                        tiling_state=None,
                        )
                             }
                    }
                },
            "Dense_0": {
                "AqtDotGeneral_0": {
                    "qrhs": {
                        "frozen": aqt_tensor.QTensor(
                            # The weight shape was (3136, 256) before tiling.
                            # After tiling it is (2, 1568, 256).
                            # Contraction shape 3136 is tiled to (2, 1568).
                            # The remaining shape 256 is not tiled.
                            qvalue=(expected_dtype, (2, 1568, 256)),
                            scale=[(dtype("float32"), (2, 1, 256))],
                            # The scale_t shape was (1, 256) before tiling.
                            # After tiling the scale shape is (2, 1, 256),
                            # then transposed to (2, 1, 256).
                            scale_t=None,
                            bias=[],
                            dequant_dtype=dtype("float32"),
                            tiling_state=None,
                        )
                    }
                }
            },
            "Dense_1": {
                "AqtDotGeneral_0": {
                    "qrhs": {
                        "frozen": aqt_tensor.QTensor(
                            # The weight shape was (256, 10) before tiling.
                            # After tiling it is (2, 128, 10).
                            # Contraction shape 256 is tiled to (2, 128).
                            # The remaining shape 10 is not tiled.
                            qvalue=(expected_dtype, (2, 128, 10)),
                            scale=[(dtype("float32"), (2, 1, 10))],
                            # The scale_t shape was (1, 10) before tiling.
                            # After tiling the scale shape is (2, 1, 10),
                            # then transposed to (2, 1, 10).
                            scale_t=None,
                            bias=[],
                            dequant_dtype=dtype("float32"),
                            tiling_state=None,
                        )
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
    model_serving = zero_out_params(model_serving, "Conv_0")
    model_serving = zero_out_params(model_serving, "Conv_1")

    # Stage 3: inference mode.
    logits_s3, _ = forward(model_serving, apply_serving)
    assert (logits_s3 == logits_s1).all()

    # Sanity check 2: Frozen weights are indeed used for inference.
    #   If we zero them out, loss would change.
    qt = model_serving[
        "aqt"
        ]["Conv_0"]["AqtConvGeneralDilated_0"]["qrhs"]["frozen"]
    qt.qvalue = jnp.zeros(qt.qvalue.shape).astype(qt.qvalue.dtype)
    bad_logits, _ = forward(model_serving, apply_serving)
    assert not (bad_logits == logits_s3).all()

  @parameterized.parameters([
      (
          {
              "drhs_bits": 8,
              "drhs_accumulator_dtype": jnp.int32,  # overwrite the default None
          },
          8,
      ),
      (
          {
              "fwd_bits": 4,
              "fwd_accumulator_dtype": None,
              "dlhs_accumulator_dtype": None,
          },
          4,
      ),
  ])
  def test_mnist_calibration(self, configs, bits):
    aqt_cfg_dg = config.config_v4(**configs)
    device_kind = jax.devices()[0].device_kind
    if device_kind == "cpu" and bits == 4:
      # Some 4-bit operations are not supported on cpu.
      # Omitting tests on cpu with 4-bits.
      return

    # RNGs
    rng = jax.random.key(0)
    rng, init_rng = jax.random.split(rng)
    rng, image_rng1, image_rng2 = jax.random.split(rng, 3)
    rng, label_rng1, label_rng2 = jax.random.split(rng, 3)
    rng, input_rng = jax.random.split(rng)
    rng, calibration_rng = jax.random.split(rng)
    del rng

    # Dataset
    ds_size = 64
    batch_size = 8
    ds = _dummy_dataset(ds_size, image_rng1, label_rng1)
    ds2 = _dummy_dataset(ds_size, image_rng2, label_rng2)

    # Stage 1: regular training
    state = flax_e2e_model.create_train_state(init_rng, aqt_cfg_dg=aqt_cfg_dg)

    state, _, _ = flax_e2e_model.train_epoch(
        state, ds, batch_size, rng=input_rng
    )

    # Stage 2: Calibration.
    flax_e2e_model.update_cfg_with_calibration(state.cnn_train.aqt_cfg_dg)
    flax_e2e_model.update_cfg_with_calibration(state.cnn_eval.aqt_cfg_dg)
    calibrate_f, model_calibrate = flax_e2e_model.calibration_conversion(state)

    calibration_steps = 4
    calibrated_params = flax_e2e_model.calibrate_epoch(
        calibrate_f,
        model_calibrate,
        ds,
        batch_size,
        rng=calibration_rng,
        calibration_steps=calibration_steps,
    )
    calibration_pytree = jax.tree_util.tree_map(
        lambda x: (x.dtype, x.shape), calibrated_params
    )
    dtype = jnp.dtype
    expected_calibration_pytree = {
        "AqtEinsum_0": {
            # For the Einsum case, lhs and rhs are swapped.
            "AqtDotGeneral_0": {
                "MeanOfAbsMaxCalibration_0": {
                    "count": (dtype("int32"), ()),
                    "sum_of_max": (dtype("float32"), (1, 1, 1)),
                },
                "MeanOfAbsMaxCalibration_1": {
                    "count": (dtype("int32"), ()),
                    "sum_of_max": (dtype("float32"), (2, 1, 10)),
                },
            }
        },
        "Dense_0": {
            "AqtDotGeneral_0": {
                "MeanOfAbsMaxCalibration_0": {
                    "count": (dtype("int32"), ()),
                    "sum_of_max": (dtype("float32"), (1, 1, 1)),
                },
                "MeanOfAbsMaxCalibration_1": {
                    "count": (dtype("int32"), ()),
                    "sum_of_max": (dtype("float32"), (2, 1, 256)),
                },
            }
        },
        "Dense_1": {
            "AqtDotGeneral_0": {
                "MeanOfAbsMaxCalibration_0": {
                    "count": (dtype("int32"), ()),
                    "sum_of_max": (dtype("float32"), (1, 1, 1)),
                },
                "MeanOfAbsMaxCalibration_1": {
                    "count": (dtype("int32"), ()),
                    "sum_of_max": (dtype("float32"), (2, 1, 10)),
                },
            }
        },
    }
    utils.test_pprint_eq(expected_calibration_pytree, calibration_pytree["qc"])

    # The count number should be equal to the number of calibration.
    einsum_params = calibrated_params["qc"]["AqtEinsum_0"]["AqtDotGeneral_0"]
    einsum_count = einsum_params["MeanOfAbsMaxCalibration_0"]["count"]
    self.assertEqual(calibration_steps, einsum_count)

    # Stage 3: Training with the calibrated numbers.
    state = state.replace(model=copy.deepcopy(calibrated_params))
    state, _, _ = flax_e2e_model.train_epoch(
        state, ds2, batch_size, rng=input_rng
    )

    # The calibrated parameters must not change.
    jax.tree.map(
        np.testing.assert_array_equal,
        calibrated_params["qc"],
        state.model["qc"],
    )

    # Other parameters should change due to the training.
    def assert_array_not_equal(x, y):
      mean_err = jnp.mean(jnp.abs(x - y))
      if mean_err == 0.0:
        assert False

    jax.tree.map(
        assert_array_not_equal,
        calibrated_params["params"],
        state.model["params"],
    )

    # Stage 4. Convert the calibrated checkpoint.
    serve_fn, model_serving = flax_e2e_model.serving_conversion(
        state, weight_only=False
    )
    dtype = jnp.dtype
    expected_dtype = dtype("int4") if bits == 4 else dtype("int8")
    expected_aqt_pytree = {
        "AqtEinsum_0": {
            "AqtDotGeneral_0": {
                "qlhs": {
                    "frozen": aqt_tensor.QTensor(
                        qvalue=(expected_dtype, (2, 5, 10)),
                        scale=[(dtype("float32"), (2, 1, 10))],
                        scale_t=None,
                        bias=[],
                        dequant_dtype=dtype("float32"),
                    )
                },
                "qrhs": {
                    "frozen": aqt_tensor.QTensor(
                        qvalue=None,
                        scale=[(dtype("float32"), (1, 1, 1))],
                        scale_t=None,
                        bias=[],
                        dequant_dtype=dtype("float32"),
                    )
                },
            }
        },
        "Dense_0": {
            "AqtDotGeneral_0": {
                "qlhs": {
                    "frozen": aqt_tensor.QTensor(
                        qvalue=None,
                        scale=[(dtype("float32"), (1, 1, 1))],
                        scale_t=None,
                        bias=[],
                        dequant_dtype=dtype("float32"),
                    )
                },
                "qrhs": {
                    "frozen": aqt_tensor.QTensor(
                        qvalue=(expected_dtype, (2, 1568, 256)),
                        scale=[(dtype("float32"), (2, 1, 256))],
                        scale_t=None,
                        bias=[],
                        dequant_dtype=dtype("float32"),
                    )
                },
            }
        },
        "Dense_1": {
            "AqtDotGeneral_0": {
                "qlhs": {
                    "frozen": aqt_tensor.QTensor(
                        qvalue=None,
                        scale=[(dtype("float32"), (1, 1, 1))],
                        scale_t=None,
                        bias=[],
                        dequant_dtype=dtype("float32"),
                    )
                },
                "qrhs": {
                    "frozen": aqt_tensor.QTensor(
                        qvalue=(expected_dtype, (2, 128, 10)),
                        scale=[(dtype("float32"), (2, 1, 10))],
                        scale_t=None,
                        bias=[],
                        dequant_dtype=dtype("float32"),
                    )
                },
            }
        },
    }

    serving_pytree = jax.tree_util.tree_map(
        lambda x: (x.dtype, x.shape), model_serving
    )
    utils.test_pprint_eq(expected_aqt_pytree, serving_pytree["aqt"])

    # Compare logits of models before conversion and after conversion.
    def forward(model, apply_fn):
      return apply_fn(
          model,
          ds["image"],
          rngs={"params": jax.random.PRNGKey(0)},
          mutable=True,
      )

    logits_before_conversion, _ = forward(state.model, state.cnn_eval.apply)
    logits_after_conversion, _ = forward(model_serving, serve_fn)
    assert (logits_before_conversion == logits_after_conversion).all()

  @parameterized.parameters([
      (
          {
              "drhs_bits": 8,
          },
          8,
      ),
      (
          {
              "fwd_bits": 4,
              "fwd_accumulator_dtype": None,
              "dlhs_accumulator_dtype": None,
          },
          4,
      ),
  ])
  def test_mnist_weighted_stats_calibration(self, configs, bits):
    aqt_cfg_dg = config.config_v4(**configs)
    device_kind = jax.devices()[0].device_kind
    if device_kind == "cpu" and bits == 4:
      # Some 4-bit operations are not supported on cpu.
      # Omitting tests on cpu with 4-bits.
      return

    # Update cfg with WeightedStatsCalibration.
    calibration_cls = functools.partial(
        aqt_flax_calibration.WeightedStatsCalibration,
        l1_dev_coeff=1.0,
        lp_dev_coeff=1.0,
        max_dev_coeff=1.0,
        const_bound_coeff=1.0,
        quant_collection="qc",
    )
    config.set_fwd_calibration(aqt_cfg_dg, calibration_cls)
    aqt_cfg_dg.fwd.dg_quantizer.lhs.calib_shared_axes = "per_tensor"

    # RNGs
    rng = jax.random.key(0)
    rng, init_rng = jax.random.split(rng)
    rng, image_rng, label_rng = jax.random.split(rng, 3)
    rng, input_rng = jax.random.split(rng)
    del rng

    # Dataset
    ds_size = 64
    batch_size = 8
    ds = _dummy_dataset(ds_size, image_rng, label_rng)

    # Stage 1: Training with collecting stats.
    state = flax_e2e_model.create_train_state(init_rng, aqt_cfg_dg=aqt_cfg_dg)

    state, _, _ = flax_e2e_model.train_epoch(
        state, ds, batch_size, rng=input_rng
    )

    trained_pytree = jax.tree_util.tree_map(
        lambda x: (x.dtype, x.shape), state.model
    )
    dtype = jnp.dtype
    expected_trained_pytree = {
        "AqtEinsum_0": {
            "AqtDotGeneral_0": {
                "WeightedStatsCalibration_0": {
                    "max_of_abs_vals": (dtype("float32"), (1, 1, 1)),
                    "sum_of_l1_vals": (dtype("float32"), (1, 1, 1)),
                    "sum_of_lp_vals": (dtype("float32"), (1, 1, 1)),
                    "sum_of_ones": (dtype("float32"), (1, 1, 1)),
                    "sum_of_vals": (dtype("float32"), (1, 1, 1)),
                },
                "WeightedStatsCalibration_1": {
                    "max_of_abs_vals": (dtype("float32"), (2, 1, 10)),
                    "sum_of_l1_vals": (dtype("float32"), (2, 1, 10)),
                    "sum_of_lp_vals": (dtype("float32"), (2, 1, 10)),
                    "sum_of_ones": (dtype("float32"), (2, 1, 10)),
                    "sum_of_vals": (dtype("float32"), (2, 1, 10)),
                },
            }
        },
        "Dense_0": {
            "AqtDotGeneral_0": {
                "WeightedStatsCalibration_0": {
                    "max_of_abs_vals": (dtype("float32"), (1, 1, 1)),
                    "sum_of_l1_vals": (dtype("float32"), (1, 1, 1)),
                    "sum_of_lp_vals": (dtype("float32"), (1, 1, 1)),
                    "sum_of_ones": (dtype("float32"), (1, 1, 1)),
                    "sum_of_vals": (dtype("float32"), (1, 1, 1)),
                },
                "WeightedStatsCalibration_1": {
                    "max_of_abs_vals": (dtype("float32"), (2, 1, 256)),
                    "sum_of_l1_vals": (dtype("float32"), (2, 1, 256)),
                    "sum_of_lp_vals": (dtype("float32"), (2, 1, 256)),
                    "sum_of_ones": (dtype("float32"), (2, 1, 256)),
                    "sum_of_vals": (dtype("float32"), (2, 1, 256)),
                },
            }
        },
        "Dense_1": {
            "AqtDotGeneral_0": {
                "WeightedStatsCalibration_0": {
                    "max_of_abs_vals": (dtype("float32"), (1, 1, 1)),
                    "sum_of_l1_vals": (dtype("float32"), (1, 1, 1)),
                    "sum_of_lp_vals": (dtype("float32"), (1, 1, 1)),
                    "sum_of_ones": (dtype("float32"), (1, 1, 1)),
                    "sum_of_vals": (dtype("float32"), (1, 1, 1)),
                },
                "WeightedStatsCalibration_1": {
                    "max_of_abs_vals": (dtype("float32"), (2, 1, 10)),
                    "sum_of_l1_vals": (dtype("float32"), (2, 1, 10)),
                    "sum_of_lp_vals": (dtype("float32"), (2, 1, 10)),
                    "sum_of_ones": (dtype("float32"), (2, 1, 10)),
                    "sum_of_vals": (dtype("float32"), (2, 1, 10)),
                },
            }
        },
    }

    utils.test_pprint_eq(expected_trained_pytree, trained_pytree["qc"])

    def forward(model, apply_fn):
      return apply_fn(
          model,
          ds["image"],
          rngs={"params": jax.random.PRNGKey(0)},
          mutable=True,
      )

    logits_before_conversion, params = forward(
        state.model, state.cnn_eval.apply
    )
    state = state.replace(model=params)

    # Stage 2. Convert the checkpoint, and serve.
    serve_fn, model_serving = flax_e2e_model.serving_conversion(
        state, weight_only=False
    )
    dtype = jnp.dtype
    expected_dtype = dtype("int4") if bits == 4 else dtype("int8")
    expected_aqt_pytree = {
        "AqtEinsum_0": {
            "AqtDotGeneral_0": {
                "qlhs": {
                    "frozen": aqt_tensor.QTensor(
                        qvalue=(expected_dtype, (2, 5, 10)),
                        scale=[(dtype("float32"), (2, 1, 10))],
                        scale_t=None,
                        bias=[],
                        dequant_dtype=dtype("float32"),
                    )
                },
                "qrhs": {
                    "frozen": aqt_tensor.QTensor(
                        qvalue=None,
                        scale=[(dtype("float32"), (1, 1, 1))],
                        scale_t=None,
                        bias=[],
                        dequant_dtype=dtype("float32"),
                    )
                },
            }
        },
        "Dense_0": {
            "AqtDotGeneral_0": {
                "qlhs": {
                    "frozen": aqt_tensor.QTensor(
                        qvalue=None,
                        scale=[(dtype("float32"), (1, 1, 1))],
                        scale_t=None,
                        bias=[],
                        dequant_dtype=dtype("float32"),
                    )
                },
                "qrhs": {
                    "frozen": aqt_tensor.QTensor(
                        qvalue=(expected_dtype, (2, 1568, 256)),
                        scale=[(dtype("float32"), (2, 1, 256))],
                        scale_t=None,
                        bias=[],
                        dequant_dtype=dtype("float32"),
                    )
                },
            }
        },
        "Dense_1": {
            "AqtDotGeneral_0": {
                "qlhs": {
                    "frozen": aqt_tensor.QTensor(
                        qvalue=None,
                        scale=[(dtype("float32"), (1, 1, 1))],
                        scale_t=None,
                        bias=[],
                        dequant_dtype=dtype("float32"),
                    )
                },
                "qrhs": {
                    "frozen": aqt_tensor.QTensor(
                        qvalue=(expected_dtype, (2, 128, 10)),
                        scale=[(dtype("float32"), (2, 1, 10))],
                        scale_t=None,
                        bias=[],
                        dequant_dtype=dtype("float32"),
                    )
                },
            }
        },
    }

    serving_pytree = jax.tree_util.tree_map(
        lambda x: (x.dtype, x.shape), model_serving
    )
    utils.test_pprint_eq(expected_aqt_pytree, serving_pytree["aqt"])

    # Compare logits of models before conversion and after conversion.
    logits_after_conversion, _ = forward(model_serving, serve_fn)
    assert (logits_before_conversion == logits_after_conversion).all()

  @parameterized.parameters(
      (["e4m3"] * 2 + ["e5m2"] * 4,),  # Higher precision fwd, larger range bwd
      ([8] * 6,),
  )
  def test_mnist_delayed_scaling_calibration(self, bits):
    # RNGs
    rng = jax.random.key(0)
    rng, init_rng = jax.random.split(rng)
    rng, image_rng, label_rng = jax.random.split(rng, 3)
    rng, input_rng = jax.random.split(rng)
    del rng

    # Dataset
    ds_size = 64
    batch_size = 8
    ds = _dummy_dataset(ds_size, image_rng, label_rng)
    num_train_steps = ds_size // batch_size

    # AQT Config
    aqt_cfg_dg = config.config_v4()
    aqt_cfg_dg = config.set_bits(aqt_cfg_dg, *bits)

    # Update cfg with DelayedScalingCalibration.
    # Setting history length to be greater than the number of training steps
    # so we can verify history is being updated correctly on each step below.
    amax_history_length = num_train_steps * 2
    calibration_cls = functools.partial(
        delayed_scaling_calibration.DelayedScalingCalibration,
        amax_history_length=amax_history_length,
    )
    aqt_cfg_dg.fwd.dg_quantizer.lhs.calibration = calibration_cls
    aqt_cfg_dg.fwd.dg_quantizer.rhs.calibration = calibration_cls
    aqt_cfg_dg.dlhs.dg_quantizer.lhs.calibration = calibration_cls
    aqt_cfg_dg.dlhs.dg_quantizer.rhs.calibration = calibration_cls
    aqt_cfg_dg.drhs.dg_quantizer.lhs.calibration = calibration_cls
    aqt_cfg_dg.drhs.dg_quantizer.rhs.calibration = calibration_cls

    # Stage 1: Training with Delayed Scaling Calibration
    state = flax_e2e_model.create_train_state(init_rng, aqt_cfg_dg=aqt_cfg_dg)

    state, _, _ = flax_e2e_model.train_epoch(
        state, ds, batch_size, rng=input_rng
    )

    trained_pytree = jax.tree_util.tree_map(
        lambda x: (x.dtype, x.shape),
        state.model[delayed_scaling_calibration.CALIBRATION_STATS],
    )
    dtype = jnp.dtype
    expected_trained_pytree = {
        "AqtEinsum_0": {
            "AqtDotGeneral_0": {
                "DelayedScalingCalibration_0": {
                    "amax_history": (dtype("float32"), (amax_history_length,)),
                    "bound": (dtype("float32"), (1,)),
                },
                "DelayedScalingCalibration_1": {
                    "amax_history": (dtype("float32"), (amax_history_length,)),
                    "bound": (dtype("float32"), (1,)),
                },
                "DelayedScalingCalibration_2": {
                    "amax_history": (dtype("float32"), (amax_history_length,)),
                    "bound": (dtype("float32"), (1,)),
                },
                "DelayedScalingCalibration_3": {
                    "amax_history": (dtype("float32"), (amax_history_length,)),
                    "bound": (dtype("float32"), (1,)),
                },
                "DelayedScalingCalibration_4": {
                    "amax_history": (dtype("float32"), (amax_history_length,)),
                    "bound": (dtype("float32"), (1,)),
                },
                "DelayedScalingCalibration_5": {
                    "amax_history": (dtype("float32"), (amax_history_length,)),
                    "bound": (dtype("float32"), (1,)),
                },
            }
        },
        "Dense_0": {
            "AqtDotGeneral_0": {
                "DelayedScalingCalibration_0": {
                    "amax_history": (dtype("float32"), (amax_history_length,)),
                    "bound": (dtype("float32"), (1,)),
                },
                "DelayedScalingCalibration_1": {
                    "amax_history": (dtype("float32"), (amax_history_length,)),
                    "bound": (dtype("float32"), (1,)),
                },
                "DelayedScalingCalibration_2": {
                    "amax_history": (dtype("float32"), (amax_history_length,)),
                    "bound": (dtype("float32"), (1,)),
                },
                "DelayedScalingCalibration_3": {
                    "amax_history": (dtype("float32"), (amax_history_length,)),
                    "bound": (dtype("float32"), (1,)),
                },
                "DelayedScalingCalibration_4": {
                    "amax_history": (dtype("float32"), (amax_history_length,)),
                    "bound": (dtype("float32"), (1,)),
                },
                "DelayedScalingCalibration_5": {
                    "amax_history": (dtype("float32"), (amax_history_length,)),
                    "bound": (dtype("float32"), (1,)),
                },
            }
        },
        "Dense_1": {
            "AqtDotGeneral_0": {
                "DelayedScalingCalibration_0": {
                    "amax_history": (dtype("float32"), (amax_history_length,)),
                    "bound": (dtype("float32"), (1,)),
                },
                "DelayedScalingCalibration_1": {
                    "amax_history": (dtype("float32"), (amax_history_length,)),
                    "bound": (dtype("float32"), (1,)),
                },
                "DelayedScalingCalibration_2": {
                    "amax_history": (dtype("float32"), (amax_history_length,)),
                    "bound": (dtype("float32"), (1,)),
                },
                "DelayedScalingCalibration_3": {
                    "amax_history": (dtype("float32"), (amax_history_length,)),
                    "bound": (dtype("float32"), (1,)),
                },
                "DelayedScalingCalibration_4": {
                    "amax_history": (dtype("float32"), (amax_history_length,)),
                    "bound": (dtype("float32"), (1,)),
                },
                "DelayedScalingCalibration_5": {
                    "amax_history": (dtype("float32"), (amax_history_length,)),
                    "bound": (dtype("float32"), (1,)),
                },
            }
        },
    }

    utils.test_pprint_eq(expected_trained_pytree, trained_pytree)

    # Verify we get the number of updates we expect to amax_history. This is a
    # bit difficult to verify since sometimes the amax_update is 0, but a few of
    # of the modules have more predictable behavior

    # We should have num_train_steps + 1 (for init) updates for the forward pass
    self.assertEqual(
        jnp.sum(
            state.model[delayed_scaling_calibration.CALIBRATION_STATS][
                "Dense_0"
            ]["AqtDotGeneral_0"]["DelayedScalingCalibration_0"]["amax_history"][
                :
            ]
            != 0
        ).item(),
        num_train_steps + 1,
    )
    self.assertNotEqual(
        state.model[delayed_scaling_calibration.CALIBRATION_STATS]["Dense_0"][
            "AqtDotGeneral_0"
        ]["DelayedScalingCalibration_0"]["bound"][:].item(),
        1.0,
    )

    # We should have num_train_steps updates for the backward pass
    self.assertEqual(
        jnp.sum(
            state.model[delayed_scaling_calibration.CALIBRATION_STATS][
                "Dense_0"
            ]["AqtDotGeneral_0"]["DelayedScalingCalibration_3"]["amax_history"][
                :
            ]
            != 0
        ).item(),
        num_train_steps,
    )
    self.assertNotEqual(
        state.model[delayed_scaling_calibration.CALIBRATION_STATS]["Dense_0"][
            "AqtDotGeneral_0"
        ]["DelayedScalingCalibration_3"]["bound"][:].item(),
        1.0,
    )

    def forward(model, apply_fn):
      return apply_fn(
          model,
          ds["image"],
          rngs={"params": jax.random.PRNGKey(0)},
          mutable=True,
      )

    # We need to set the mode to SERVE so the MutableArrays don't get changed,
    # which would mean the logits would not match before and after conversion.
    state.cnn_eval.weights_quant_mode = utils.QuantMode.SERVE
    state.cnn_eval.activations_quant_mode = utils.QuantMode.SERVE
    logits_before_conversion, params = forward(
        state.model, state.cnn_eval.apply
    )
    state = state.replace(model=params)

    # Stage 2. Convert the checkpoint, and serve.
    serve_fn, model_serving = flax_e2e_model.serving_conversion(
        state, weight_only=False
    )
    dtype = jnp.dtype
    expected_dtype = dtype("int8") if bits[0] == 8 else dtype("float8_e4m3fn")
    expected_aqt_pytree = {
        "AqtEinsum_0": {
            "AqtDotGeneral_0": {
                "qlhs": {
                    "frozen": aqt_tensor.QTensor(
                        qvalue=(expected_dtype, (2, 5, 10)),
                        scale=[(dtype("float32"), (1, 1, 1))],
                        scale_t=None,
                        dequant_dtype=dtype("float32"),
                        bias=[],
                    )
                },
                "qrhs": {
                    "frozen": aqt_tensor.QTensor(
                        qvalue=None,
                        scale=[(dtype("float32"), (1, 1, 1))],
                        scale_t=None,
                        dequant_dtype=dtype("float32"),
                        bias=[],
                    )
                },
            }
        },
        "Dense_0": {
            "AqtDotGeneral_0": {
                "qlhs": {
                    "frozen": aqt_tensor.QTensor(
                        qvalue=None,
                        scale=[(dtype("float32"), (1, 1, 1))],
                        scale_t=None,
                        dequant_dtype=dtype("float32"),
                        bias=[],
                    )
                },
                "qrhs": {
                    "frozen": aqt_tensor.QTensor(
                        qvalue=(expected_dtype, (2, 1568, 256)),
                        scale=[(dtype("float32"), (1, 1, 1))],
                        scale_t=None,
                        dequant_dtype=dtype("float32"),
                        bias=[],
                    )
                },
            }
        },
        "Dense_1": {
            "AqtDotGeneral_0": {
                "qlhs": {
                    "frozen": aqt_tensor.QTensor(
                        qvalue=None,
                        scale=[(dtype("float32"), (1, 1, 1))],
                        scale_t=None,
                        dequant_dtype=dtype("float32"),
                        bias=[],
                    )
                },
                "qrhs": {
                    "frozen": aqt_tensor.QTensor(
                        qvalue=(expected_dtype, (2, 128, 10)),
                        scale=[(dtype("float32"), (1, 1, 1))],
                        scale_t=None,
                        dequant_dtype=dtype("float32"),
                        bias=[],
                    )
                },
            }
        },
    }

    serving_pytree = jax.tree_util.tree_map(
        lambda x: (x.dtype, x.shape), model_serving
    )
    utils.test_pprint_eq(expected_aqt_pytree, serving_pytree["aqt"])

    # Compare logits of models before conversion and after conversion.
    logits_after_conversion, _ = forward(model_serving, serve_fn)
    assert (logits_before_conversion == logits_after_conversion).all()

  def test_mnist_training_backward_compatibility(self):
    aqt_cfg_dg = config.config_v4()

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
    rng, image_rng = jax.random.split(rng)
    rng, label_rng = jax.random.split(rng)
    rng, input_rng = jax.random.split(rng)
    del rng

    # Dataset
    ds_size = 8
    ds = _dummy_dataset(ds_size, image_rng, label_rng)

    # Stage 1: regular training
    state = flax_e2e_model.create_train_state(init_rng, aqt_cfg_dg=aqt_cfg_dg)

    state, _, _ = flax_e2e_model.train_epoch(
        state, ds, batch_size=ds_size // 2, rng=input_rng
    )

    # Run forward once more in the same mode to get logits for testing below.
    logits_s1, _ = forward(state.model, state.cnn_eval.apply)

    # Stage 2: Model conversion (quantized weights freezing)
    # Freeze with legacy freezer; serve with new freezer.
    apply_serving, model_serving = flax_e2e_model.serving_conversion(
        state, legacy_for_freeze=True, legacy_for_serve=False
    )

    dtype = jnp.dtype
    expected_aqt_pytree = {
        "AqtEinsum_0": {
            "AqtDotGeneral_0": {
                "qlhs": {
                    "scale": (dtype("float32"), (2, 1, 10)),
                    "value": (dtype("int8"), (2, 5, 10)),
                }
            }
        },
        "Dense_0": {
            "AqtDotGeneral_0": {
                "qrhs": {
                    "scale": (dtype("float32"), (2, 1, 256)),
                    "value": (dtype("int8"), (2, 1568, 256)),
                }
            }
        },
        "Dense_1": {
            "AqtDotGeneral_0": {
                "qrhs": {
                    "scale": (dtype("float32"), (2, 1, 10)),
                    "value": (dtype("int8"), (2, 128, 10)),
                }
            }
        },
    }

    serving_pytree = jax.tree_util.tree_map(
        lambda x: (x.dtype, x.shape), model_serving
    )
    utils.test_pprint_eq(expected_aqt_pytree, serving_pytree["aqt"])

    # Stage 3: inference mode.
    logits_s3, _ = forward(model_serving, apply_serving)
    assert (logits_s3 == logits_s1).all()


if __name__ == "__main__":
  absltest.main()
