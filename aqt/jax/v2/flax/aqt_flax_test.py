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

"""Test for AQT flax."""
import difflib
import functools
import pprint
from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax.v2 import config
from aqt.jax.v2.flax import aqt_flax
import flax.linen as nn
import jax
import jax.numpy as jnp


class AqtFlaxTest(parameterized.TestCase):

  def test_aqt_einsum(self):
    class Model(nn.Module):
      aqt_cfg: config.DotGeneral | None
      lhs_qt_external: bool = False
      rhs_qt_external: bool = False

      @nn.compact
      def __call__(self, lhs, rhs):
        lhs_in = lhs
        assert not (
            (self.lhs_qt_external or self.rhs_qt_external)
            and (self.aqt_cfg is None)
        ), 'aqt_cfg cannot be None when providing qtensor as inputs to einsum'
        if self.lhs_qt_external:
          lhs_in, _ = self.aqt_cfg.fwd.lhs.quantizer.quant(
              lhs, calibration_axes=(2, 3)
          )
          lhs_dtype = self.aqt_cfg.fwd.lhs.quantizer.numerics.get_dtype()
          lhs_in = lhs_in.qvalue_astype(lhs_dtype)
        rhs_in = rhs
        if self.rhs_qt_external:
          rhs_in, _ = self.aqt_cfg.fwd.rhs.quantizer.quant(
              rhs, calibration_axes=(1, 2)
          )
          rhs_dtype = self.aqt_cfg.fwd.rhs.quantizer.numerics.get_dtype()
          rhs_in = rhs_in.qvalue_astype(rhs_dtype)
        einsum = aqt_flax.AqtEinsum(cfg=self.aqt_cfg)
        # xhs_qt can be inputs to AqtEinsum
        # xhs->xhs_qt can happen outside of AqtEinsum, e.g., k/v cache quant
        # input xhs_qt will force get_tensor() to always return xhs_qt
        out = einsum('ijkh,mkh->ijm', lhs_in, rhs_in)
        return out

    key = jax.random.PRNGKey(0)
    subkey1, subkey2 = jax.random.split(key, num=2)
    lhs = jax.random.uniform(subkey1, shape=(3, 4, 5, 6))
    rhs = jax.random.uniform(subkey2, shape=(2, 5, 6))

    def test(model_cls, cfg):
      model = model_cls(cfg)
      out = model.apply({}, lhs, rhs, rngs={'params': jax.random.PRNGKey(0)})
      # print(f'{out.shape=}')
      # print(f'{out=}')
      return out

    out_float = test(Model, None)
    out_int8 = test(Model, aqt_flax.config_v4())
    out_int8_lqt = test(
        functools.partial(Model, lhs_qt_external=True), aqt_flax.config_v4()
    )
    out_int8_rqt = test(
        functools.partial(Model, rhs_qt_external=True), aqt_flax.config_v4()
    )
    out_int8_qt = test(
        functools.partial(Model, lhs_qt_external=True, rhs_qt_external=True),
        aqt_flax.config_v4(),
    )

    assert (out_int8 == out_int8_lqt).all(), 'lhs external qt failed'
    assert (out_int8 == out_int8_rqt).all(), 'rhs external qt failed'
    assert (out_int8 == out_int8_qt).all(), 'both external qt failed'
    mse = jnp.mean(jnp.square(out_int8_qt - out_float))
    assert mse > 0, 'Mean square error is 0. Einsum is not quantized.'

  def test_einsum_grad_leak(self):
    class CNN(nn.Module):
      aqt_cfg: config.DotGeneral

      @nn.compact
      def __call__(self, x):
        einsum = aqt_flax.AqtEinsum(self.aqt_cfg)
        x = einsum('bc,ab->ac', jnp.identity(10, dtype=x.dtype), x)
        return x

    model = CNN(aqt_cfg=config.fully_quantized())
    var = model.init({'params': jax.random.PRNGKey(0)}, jnp.ones(shape=(1, 10)))

    @jax.jit
    def apply_fn(inputs):
      return model.apply(
          var, inputs, rngs={'params': jax.random.PRNGKey(0)}, mutable=True
      )

    @jax.jit
    @jax.value_and_grad
    def train_step(inputs):
      return jnp.sum(apply_fn(inputs)[0])

    train_step(jnp.ones(shape=(1, 10)))

  def test_config_v4(self):
    cfg = aqt_flax.config_v4(
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
    cfg_str = pprint.pformat(cfg)
    expected_cfg_str = """DotGeneral(fwd=DotGeneralRaw(lhs=Tensor(quantizer=Quantizer(numerics=IntNumerics(bits=8,
                                                                                 preserve_zero=True,
                                                                                 preserve_max_val=False,
                                                                                 clip=True,
                                                                                 clip_gradient=False,
                                                                                 round=True,
                                                                                 noise_fn=None,
                                                                                 dtype=<class 'jax.numpy.int8'>),
                                                            calib_shared_axes=None,
                                                            scale_stop_grad=True,
                                                            calibration=AbsMaxCalibration(),
                                                            po2_scale=False,
                                                            context=Context(key=None,
                                                                            train_step=None)),
                                        use_fwd_quant=None,
                                        dequant_mode=<DequantMode.OUTPUT: 1>),
                             rhs=Tensor(quantizer=Quantizer(numerics=IntNumerics(bits=8,
                                                                                 preserve_zero=True,
                                                                                 preserve_max_val=False,
                                                                                 clip=True,
                                                                                 clip_gradient=False,
                                                                                 round=True,
                                                                                 noise_fn=None,
                                                                                 dtype=<class 'jax.numpy.int8'>),
                                                            calib_shared_axes=None,
                                                            scale_stop_grad=True,
                                                            calibration=AbsMaxCalibration(),
                                                            po2_scale=False,
                                                            context=Context(key=None,
                                                                            train_step=None)),
                                        use_fwd_quant=None,
                                        dequant_mode=<DequantMode.OUTPUT: 1>),
                             dg_accumulator_dtype=<class 'jax.numpy.int16'>,
                             local_aqt=None,
                             jax_scope_name='aqt_fwd'),
           dlhs=DotGeneralRaw(lhs=Tensor(quantizer=Quantizer(numerics=IntNumerics(bits=7,
                                                                                  preserve_zero=True,
                                                                                  preserve_max_val=False,
                                                                                  clip=True,
                                                                                  clip_gradient=False,
                                                                                  round=True,
                                                                                  noise_fn=RandomCenteredUniform(),
                                                                                  dtype=<class 'jax.numpy.int8'>),
                                                             calib_shared_axes=None,
                                                             scale_stop_grad=True,
                                                             calibration=AbsMaxCalibration(),
                                                             po2_scale=False,
                                                             context=Context(key=None,
                                                                             train_step=None)),
                                         use_fwd_quant=None,
                                         dequant_mode=<DequantMode.OUTPUT: 1>),
                              rhs=Tensor(quantizer=Quantizer(numerics=IntNumerics(bits=7,
                                                                                  preserve_zero=True,
                                                                                  preserve_max_val=False,
                                                                                  clip=True,
                                                                                  clip_gradient=False,
                                                                                  round=True,
                                                                                  noise_fn=None,
                                                                                  dtype=<class 'jax.numpy.int8'>),
                                                             calib_shared_axes=None,
                                                             scale_stop_grad=True,
                                                             calibration=AbsMaxCalibration(),
                                                             po2_scale=False,
                                                             context=Context(key=None,
                                                                             train_step=None)),
                                         use_fwd_quant=False,
                                         dequant_mode=<DequantMode.OUTPUT: 1>),
                              dg_accumulator_dtype=<class 'jax.numpy.int8'>,
                              local_aqt=LocalAqt(contraction_axis_shard_count=2),
                              jax_scope_name='aqt_dlhs'),
           drhs=DotGeneralRaw(lhs=Tensor(quantizer=Quantizer(numerics=IntNumerics(bits=6,
                                                                                  preserve_zero=True,
                                                                                  preserve_max_val=False,
                                                                                  clip=True,
                                                                                  clip_gradient=False,
                                                                                  round=True,
                                                                                  noise_fn=RandomCenteredUniform(),
                                                                                  dtype=<class 'jax.numpy.int8'>),
                                                             calib_shared_axes=None,
                                                             scale_stop_grad=True,
                                                             calibration=AbsMaxCalibration(),
                                                             po2_scale=False,
                                                             context=Context(key=None,
                                                                             train_step=None)),
                                         use_fwd_quant=None,
                                         dequant_mode=<DequantMode.OUTPUT: 1>),
                              rhs=Tensor(quantizer=Quantizer(numerics=IntNumerics(bits=6,
                                                                                  preserve_zero=True,
                                                                                  preserve_max_val=False,
                                                                                  clip=True,
                                                                                  clip_gradient=False,
                                                                                  round=True,
                                                                                  noise_fn=None,
                                                                                  dtype=<class 'jax.numpy.int8'>),
                                                             calib_shared_axes=None,
                                                             scale_stop_grad=True,
                                                             calibration=AbsMaxCalibration(),
                                                             po2_scale=False,
                                                             context=Context(key=None,
                                                                             train_step=None)),
                                         use_fwd_quant=False,
                                         dequant_mode=<DequantMode.OUTPUT: 1>),
                              dg_accumulator_dtype=<class 'jax.numpy.int4'>,
                              local_aqt=LocalAqt(contraction_axis_shard_count=3),
                              jax_scope_name='aqt_drhs'))"""

    def print_diff():
      diff_generator = difflib.context_diff(
          cfg_str.split(' '), expected_cfg_str.split(' ')
      )
      for diff in diff_generator:
        print(diff)

    assert cfg_str == expected_cfg_str, print_diff()


if __name__ == '__main__':
  absltest.main()
