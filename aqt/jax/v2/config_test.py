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

"""Test for AQT configs."""
from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax.v2 import config
from aqt.jax.v2 import utils
import jax.numpy as jnp


class AqtConfigTest(parameterized.TestCase):

  def _retrieve_quantizers(self, dot_general_raws):
    ret = []
    for dot_general_raw in dot_general_raws:
      ret.extend(
          [dot_general_raw.dg_quantizer.lhs, dot_general_raw.dg_quantizer.rhs]
      )
    return ret

  def test_config_v4(self):
    cfg = config.config_v4(
        fwd_bits=8,
        dlhs_bits=7,
        drhs_bits=6,
        rng_type='custom-1',
        dlhs_local_aqt=config.LocalAqt(2),
        drhs_local_aqt=config.LocalAqt(3),
        fwd_accumulator_dtype=jnp.int16,
        dlhs_accumulator_dtype=jnp.int8,
        drhs_accumulator_dtype=jnp.int4,
    )
    expected_cfg_str = """DotGeneral(fwd=DotGeneralRaw(lhs=Tensor(use_fwd_quant=None,
                                        dequant_mode=<DequantMode.OUTPUT: 1>,
                                        calibration_mode=<CalibrationMode.CONTRACTING_AXIS: 1>),
                             rhs=Tensor(use_fwd_quant=None,
                                        dequant_mode=<DequantMode.OUTPUT: 1>,
                                        calibration_mode=<CalibrationMode.CONTRACTING_AXIS: 1>),
                             dg_quantizer=DefaultDotGeneralQuantizer(lhs=Quantizer(numerics=IntNumerics(bits=8,
                                                                                                        preserve_zero=True,
                                                                                                        preserve_max_val=False,
                                                                                                        clip=True,
                                                                                                        clip_gradient=False,
                                                                                                        round=True,
                                                                                                        noise_fn=None,
                                                                                                        dtype=<class 'jax.numpy.int8'>),
                                                                                   calib_shared_axes=None,
                                                                                   scale_stop_grad=True,
                                                                                   calibration=AbsMaxCalibration(scale=None),
                                                                                   po2_scale=False,
                                                                                   context=Context(key=None,
                                                                                                   train_step=None)),
                                                                     rhs=Quantizer(numerics=IntNumerics(bits=8,
                                                                                                        preserve_zero=True,
                                                                                                        preserve_max_val=False,
                                                                                                        clip=True,
                                                                                                        clip_gradient=False,
                                                                                                        round=True,
                                                                                                        noise_fn=None,
                                                                                                        dtype=<class 'jax.numpy.int8'>),
                                                                                   calib_shared_axes=None,
                                                                                   scale_stop_grad=True,
                                                                                   calibration=AbsMaxCalibration(scale=None),
                                                                                   po2_scale=False,
                                                                                   context=Context(key=None,
                                                                                                   train_step=None))),
                             dg_accumulator_dtype=<class 'jax.numpy.int16'>,
                             local_aqt=None,
                             jax_scope_name='aqt_fwd',
                             allow_dummy_gradient_into_qtensor=False,
                             dot_general=<function dot_general>),
           dlhs=DotGeneralRaw(lhs=Tensor(use_fwd_quant=None,
                                         dequant_mode=<DequantMode.OUTPUT: 1>,
                                         calibration_mode=<CalibrationMode.CONTRACTING_AXIS: 1>),
                              rhs=Tensor(use_fwd_quant=False,
                                         dequant_mode=<DequantMode.OUTPUT: 1>,
                                         calibration_mode=<CalibrationMode.CONTRACTING_AXIS: 1>),
                              dg_quantizer=DefaultDotGeneralQuantizer(lhs=Quantizer(numerics=IntNumerics(bits=7,
                                                                                                         preserve_zero=True,
                                                                                                         preserve_max_val=False,
                                                                                                         clip=True,
                                                                                                         clip_gradient=False,
                                                                                                         round=True,
                                                                                                         noise_fn=RandomCenteredUniform(),
                                                                                                         dtype=<class 'jax.numpy.int8'>),
                                                                                    calib_shared_axes=None,
                                                                                    scale_stop_grad=True,
                                                                                    calibration=AbsMaxCalibration(scale=None),
                                                                                    po2_scale=False,
                                                                                    context=Context(key=None,
                                                                                                    train_step=None)),
                                                                      rhs=Quantizer(numerics=IntNumerics(bits=7,
                                                                                                         preserve_zero=True,
                                                                                                         preserve_max_val=False,
                                                                                                         clip=True,
                                                                                                         clip_gradient=False,
                                                                                                         round=True,
                                                                                                         noise_fn=None,
                                                                                                         dtype=<class 'jax.numpy.int8'>),
                                                                                    calib_shared_axes=None,
                                                                                    scale_stop_grad=True,
                                                                                    calibration=AbsMaxCalibration(scale=None),
                                                                                    po2_scale=False,
                                                                                    context=Context(key=None,
                                                                                                    train_step=None))),
                              dg_accumulator_dtype=<class 'jax.numpy.int8'>,
                              local_aqt=LocalAqt(contraction_axis_shard_count=2),
                              jax_scope_name='aqt_dlhs',
                              allow_dummy_gradient_into_qtensor=False,
                              dot_general=<function dot_general>),
           drhs=DotGeneralRaw(lhs=Tensor(use_fwd_quant=None,
                                         dequant_mode=<DequantMode.OUTPUT: 1>,
                                         calibration_mode=<CalibrationMode.CONTRACTING_AXIS: 1>),
                              rhs=Tensor(use_fwd_quant=False,
                                         dequant_mode=<DequantMode.OUTPUT: 1>,
                                         calibration_mode=<CalibrationMode.CONTRACTING_AXIS: 1>),
                              dg_quantizer=DefaultDotGeneralQuantizer(lhs=Quantizer(numerics=IntNumerics(bits=6,
                                                                                                         preserve_zero=True,
                                                                                                         preserve_max_val=False,
                                                                                                         clip=True,
                                                                                                         clip_gradient=False,
                                                                                                         round=True,
                                                                                                         noise_fn=RandomCenteredUniform(),
                                                                                                         dtype=<class 'jax.numpy.int8'>),
                                                                                    calib_shared_axes=None,
                                                                                    scale_stop_grad=True,
                                                                                    calibration=AbsMaxCalibration(scale=None),
                                                                                    po2_scale=False,
                                                                                    context=Context(key=None,
                                                                                                    train_step=None)),
                                                                      rhs=Quantizer(numerics=IntNumerics(bits=6,
                                                                                                         preserve_zero=True,
                                                                                                         preserve_max_val=False,
                                                                                                         clip=True,
                                                                                                         clip_gradient=False,
                                                                                                         round=True,
                                                                                                         noise_fn=None,
                                                                                                         dtype=<class 'jax.numpy.int8'>),
                                                                                    calib_shared_axes=None,
                                                                                    scale_stop_grad=True,
                                                                                    calibration=AbsMaxCalibration(scale=None),
                                                                                    po2_scale=False,
                                                                                    context=Context(key=None,
                                                                                                    train_step=None))),
                              dg_accumulator_dtype=<class 'jax.numpy.int4'>,
                              local_aqt=LocalAqt(contraction_axis_shard_count=3),
                              jax_scope_name='aqt_drhs',
                              allow_dummy_gradient_into_qtensor=False,
                              dot_general=<function dot_general>),
           apply_custom_vjp_on_jax=True)"""
    utils.test_pprint_eq(cfg, expected_cfg_str, remove_memory_addresses=True)

  def test_configv4_original(self):
    expected_cfg_str = """DotGeneral(fwd=DotGeneralRaw(lhs=Tensor(use_fwd_quant=None,
                                        dequant_mode=<DequantMode.OUTPUT: 1>,
                                        calibration_mode=<CalibrationMode.CONTRACTING_AXIS: 1>),
                             rhs=Tensor(use_fwd_quant=None,
                                        dequant_mode=<DequantMode.OUTPUT: 1>,
                                        calibration_mode=<CalibrationMode.CONTRACTING_AXIS: 1>),
                             dg_quantizer=DefaultDotGeneralQuantizer(lhs=Quantizer(numerics=IntNumerics(bits=8,
                                                                                                        preserve_zero=True,
                                                                                                        preserve_max_val=False,
                                                                                                        clip=True,
                                                                                                        clip_gradient=False,
                                                                                                        round=True,
                                                                                                        noise_fn=None,
                                                                                                        dtype=<class 'jax.numpy.int8'>),
                                                                                   calib_shared_axes=None,
                                                                                   scale_stop_grad=True,
                                                                                   calibration=AbsMaxCalibration(scale=None),
                                                                                   po2_scale=False,
                                                                                   context=Context(key=None,
                                                                                                   train_step=None)),
                                                                     rhs=Quantizer(numerics=IntNumerics(bits=8,
                                                                                                        preserve_zero=True,
                                                                                                        preserve_max_val=False,
                                                                                                        clip=True,
                                                                                                        clip_gradient=False,
                                                                                                        round=True,
                                                                                                        noise_fn=None,
                                                                                                        dtype=<class 'jax.numpy.int8'>),
                                                                                   calib_shared_axes=None,
                                                                                   scale_stop_grad=True,
                                                                                   calibration=AbsMaxCalibration(scale=None),
                                                                                   po2_scale=False,
                                                                                   context=Context(key=None,
                                                                                                   train_step=None))),
                             dg_accumulator_dtype=<class 'jax.numpy.int32'>,
                             local_aqt=None,
                             jax_scope_name='aqt_fwd',
                             allow_dummy_gradient_into_qtensor=False,
                             dot_general=<function dot_general>),
           dlhs=DotGeneralRaw(lhs=Tensor(use_fwd_quant=None,
                                         dequant_mode=<DequantMode.OUTPUT: 1>,
                                         calibration_mode=<CalibrationMode.CONTRACTING_AXIS: 1>),
                              rhs=Tensor(use_fwd_quant=False,
                                         dequant_mode=<DequantMode.OUTPUT: 1>,
                                         calibration_mode=<CalibrationMode.CONTRACTING_AXIS: 1>),
                              dg_quantizer=DefaultDotGeneralQuantizer(lhs=Quantizer(numerics=IntNumerics(bits=8,
                                                                                                         preserve_zero=True,
                                                                                                         preserve_max_val=False,
                                                                                                         clip=True,
                                                                                                         clip_gradient=False,
                                                                                                         round=True,
                                                                                                         noise_fn=JaxUniform(),
                                                                                                         dtype=<class 'jax.numpy.int8'>),
                                                                                    calib_shared_axes=None,
                                                                                    scale_stop_grad=True,
                                                                                    calibration=AbsMaxCalibration(scale=None),
                                                                                    po2_scale=False,
                                                                                    context=Context(key=None,
                                                                                                    train_step=None)),
                                                                      rhs=Quantizer(numerics=IntNumerics(bits=8,
                                                                                                         preserve_zero=True,
                                                                                                         preserve_max_val=False,
                                                                                                         clip=True,
                                                                                                         clip_gradient=False,
                                                                                                         round=True,
                                                                                                         noise_fn=None,
                                                                                                         dtype=<class 'jax.numpy.int8'>),
                                                                                    calib_shared_axes=None,
                                                                                    scale_stop_grad=True,
                                                                                    calibration=AbsMaxCalibration(scale=None),
                                                                                    po2_scale=False,
                                                                                    context=Context(key=None,
                                                                                                    train_step=None))),
                              dg_accumulator_dtype=<class 'jax.numpy.int32'>,
                              local_aqt=None,
                              jax_scope_name='aqt_dlhs',
                              allow_dummy_gradient_into_qtensor=False,
                              dot_general=<function dot_general>),
           drhs=DotGeneralRaw(lhs=Tensor(use_fwd_quant=None,
                                         dequant_mode=<DequantMode.OUTPUT: 1>,
                                         calibration_mode=<CalibrationMode.CONTRACTING_AXIS: 1>),
                              rhs=Tensor(use_fwd_quant=False,
                                         dequant_mode=<DequantMode.OUTPUT: 1>,
                                         calibration_mode=<CalibrationMode.CONTRACTING_AXIS: 1>),
                              dg_quantizer=DefaultDotGeneralQuantizer(lhs=Quantizer(numerics=NoNumerics(noise_fn=JaxUniform(),
                                                                                                        dtype=None),
                                                                                    calib_shared_axes=None,
                                                                                    scale_stop_grad=True,
                                                                                    calibration=AbsMaxCalibration(scale=None),
                                                                                    po2_scale=False,
                                                                                    context=Context(key=None,
                                                                                                    train_step=None)),
                                                                      rhs=Quantizer(numerics=NoNumerics(noise_fn=None,
                                                                                                        dtype=None),
                                                                                    calib_shared_axes=None,
                                                                                    scale_stop_grad=True,
                                                                                    calibration=AbsMaxCalibration(scale=None),
                                                                                    po2_scale=False,
                                                                                    context=Context(key=None,
                                                                                                    train_step=None))),
                              dg_accumulator_dtype=None,
                              local_aqt=None,
                              jax_scope_name='aqt_drhs',
                              allow_dummy_gradient_into_qtensor=False,
                              dot_general=<function dot_general>),
           apply_custom_vjp_on_jax=True)"""
    utils.test_pprint_eq(
        config.config_v4(), expected_cfg_str, remove_memory_addresses=True
    )

  def test_config_fwd_fp8(self):
    expected_cfg = """DotGeneral(fwd=DotGeneralRaw(lhs=Tensor(use_fwd_quant=None,
                                        dequant_mode=<DequantMode.OUTPUT: 1>,
                                        calibration_mode=<CalibrationMode.CONTRACTING_AXIS: 1>),
                             rhs=Tensor(use_fwd_quant=None,
                                        dequant_mode=<DequantMode.OUTPUT: 1>,
                                        calibration_mode=<CalibrationMode.CONTRACTING_AXIS: 1>),
                             dg_quantizer=DefaultDotGeneralQuantizer(lhs=Quantizer(numerics=Fp8Numerics(dtype=<class 'jax.numpy.float8_e4m3fn'>,
                                                                                                        exponent_bits=4,
                                                                                                        mantissa_bits=3,
                                                                                                        noise_fn=None),
                                                                                   calib_shared_axes=None,
                                                                                   scale_stop_grad=True,
                                                                                   calibration=AbsMaxCalibration(scale=None),
                                                                                   po2_scale=False,
                                                                                   context=Context(key=None,
                                                                                                   train_step=None)),
                                                                     rhs=Quantizer(numerics=Fp8Numerics(dtype=<class 'jax.numpy.float8_e4m3fn'>,
                                                                                                        exponent_bits=4,
                                                                                                        mantissa_bits=3,
                                                                                                        noise_fn=None),
                                                                                   calib_shared_axes=None,
                                                                                   scale_stop_grad=True,
                                                                                   calibration=AbsMaxCalibration(scale=None),
                                                                                   po2_scale=False,
                                                                                   context=Context(key=None,
                                                                                                   train_step=None))),
                             dg_accumulator_dtype=<class 'jax.numpy.float32'>,
                             local_aqt=None,
                             jax_scope_name='aqt_fwd',
                             allow_dummy_gradient_into_qtensor=False,
                             dot_general=<function dot_general>),
           dlhs=DotGeneralRaw(lhs=Tensor(use_fwd_quant=None,
                                         dequant_mode=<DequantMode.OUTPUT: 1>,
                                         calibration_mode=<CalibrationMode.CONTRACTING_AXIS: 1>),
                              rhs=Tensor(use_fwd_quant=False,
                                         dequant_mode=<DequantMode.OUTPUT: 1>,
                                         calibration_mode=<CalibrationMode.CONTRACTING_AXIS: 1>),
                              dg_quantizer=DefaultDotGeneralQuantizer(lhs=Quantizer(numerics=NoNumerics(noise_fn=None,
                                                                                                        dtype=None),
                                                                                    calib_shared_axes=None,
                                                                                    scale_stop_grad=True,
                                                                                    calibration=AbsMaxCalibration(scale=None),
                                                                                    po2_scale=False,
                                                                                    context=Context(key=None,
                                                                                                    train_step=None)),
                                                                      rhs=Quantizer(numerics=NoNumerics(noise_fn=None,
                                                                                                        dtype=None),
                                                                                    calib_shared_axes=None,
                                                                                    scale_stop_grad=True,
                                                                                    calibration=AbsMaxCalibration(scale=None),
                                                                                    po2_scale=False,
                                                                                    context=Context(key=None,
                                                                                                    train_step=None))),
                              dg_accumulator_dtype=None,
                              local_aqt=None,
                              jax_scope_name='aqt_dlhs',
                              allow_dummy_gradient_into_qtensor=False,
                              dot_general=<function dot_general>),
           drhs=DotGeneralRaw(lhs=Tensor(use_fwd_quant=None,
                                         dequant_mode=<DequantMode.OUTPUT: 1>,
                                         calibration_mode=<CalibrationMode.CONTRACTING_AXIS: 1>),
                              rhs=Tensor(use_fwd_quant=False,
                                         dequant_mode=<DequantMode.OUTPUT: 1>,
                                         calibration_mode=<CalibrationMode.CONTRACTING_AXIS: 1>),
                              dg_quantizer=DefaultDotGeneralQuantizer(lhs=Quantizer(numerics=NoNumerics(noise_fn=None,
                                                                                                        dtype=None),
                                                                                    calib_shared_axes=None,
                                                                                    scale_stop_grad=True,
                                                                                    calibration=AbsMaxCalibration(scale=None),
                                                                                    po2_scale=False,
                                                                                    context=Context(key=None,
                                                                                                    train_step=None)),
                                                                      rhs=Quantizer(numerics=NoNumerics(noise_fn=None,
                                                                                                        dtype=None),
                                                                                    calib_shared_axes=None,
                                                                                    scale_stop_grad=True,
                                                                                    calibration=AbsMaxCalibration(scale=None),
                                                                                    po2_scale=False,
                                                                                    context=Context(key=None,
                                                                                                    train_step=None))),
                              dg_accumulator_dtype=None,
                              local_aqt=None,
                              jax_scope_name='aqt_drhs',
                              allow_dummy_gradient_into_qtensor=False,
                              dot_general=<function dot_general>),
           apply_custom_vjp_on_jax=True)"""
    utils.test_pprint_eq(
        config.config_fwd_fp8(), expected_cfg, remove_memory_addresses=True
    )

  def test_set_int_numerics_preserve_zero(self):
    cfg = config.config_v4()
    for quantizer in self._retrieve_quantizers([cfg.fwd, cfg.dlhs]):
      self.assertTrue(quantizer.numerics.preserve_zero)
      self.assertEqual(quantizer.numerics.dtype, jnp.int8)

    config.set_int_numerics_preserve_zero(cfg, preserve_zero=False)
    for quantizer in self._retrieve_quantizers([cfg.fwd, cfg.dlhs]):
      self.assertFalse(quantizer.numerics.preserve_zero)
      self.assertIsNone(quantizer.numerics.dtype)

  def test_set_absmax_calib_scale(self):
    cfg = config.config_v4()
    for quantizer in self._retrieve_quantizers([cfg.fwd, cfg.dlhs, cfg.drhs]):
      self.assertIsNone(quantizer.calibration.scale)

    for quantizer in self._retrieve_quantizers([cfg.fwd, cfg.dlhs]):
      self.assertFalse(quantizer.numerics.clip_gradient)

    config.set_absmax_calib_scale(cfg, scale=3)
    for quantizer in self._retrieve_quantizers([cfg.fwd, cfg.dlhs, cfg.drhs]):
      self.assertAlmostEqual(quantizer.calibration.scale, 3)

    for quantizer in self._retrieve_quantizers([cfg.fwd, cfg.dlhs]):
      self.assertFalse(quantizer.numerics.clip_gradient)

    config.set_absmax_calib_scale(cfg, scale=0.1)
    for quantizer in self._retrieve_quantizers([cfg.fwd, cfg.dlhs]):
      self.assertTrue(quantizer.numerics.clip_gradient)


if __name__ == '__main__':
  absltest.main()
