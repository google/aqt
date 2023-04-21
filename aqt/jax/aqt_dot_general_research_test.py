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

"""Tests for dot_general."""
from absl.testing import absltest
from absl.testing import parameterized
import aqt.jax.aqt_dot_general_research as aqtr

import flax.linen.linear as fl
import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats


def get_dot_generals(f, *args):
  jaxpr = jax.make_jaxpr(f)(*args)
  for i in jaxpr.eqns:
    if i.primitive.name == "dot_general":
      [lhs, rhs] = i.invars
      [out] = i.outvars
      yield (lhs.aval, rhs.aval, out.aval)


rngkey = None


def rand_unif(shape, maxval):
  global rngkey
  if rngkey is None:
    rngkey = jax.random.PRNGKey(0)
  k1, k2 = jax.random.split(rngkey)
  rngkey = k1
  return jax.random.uniform(key=k2, shape=shape, minval=-maxval, maxval=maxval)


# The main test strategy is :
# - Test that FQ is sensible (improve fq_noise_test)
#   - Quantization noise (rounding error) is of proper value
#   - Cliping is correct
#   - Stochastic noise is added correctly.
# - Test that we really use int8/int4/bool in XLA/HLO
# - Test of fq vs dot_general equivalence in po2
# - Test that it all applies to gradients
#
#
# TODO(lew): Tests are a bit incomplete. What we need:
# On top of that we shell test:
#  - Gradinent is 1 in the range, 0 outside the abs-max range.


def test_eq(name, a, b):
  # TODO(lew): use library function.
  mean_err = jnp.mean(jnp.abs(a - b))
  if mean_err != 0.0:
    print("mean_err =", mean_err)
    print(a.shape)
    print(a[:3, :3])
    print(b.shape)
    print(b[:3, :3])
    print("fail: not equal; case: ", name)
    assert False


def check_eq(dg1, dg2, lhs, rhs, gra, lr_mult=1.0, gl_mult=1.0, gr_mult=1.0):
  """Tests whether dg1 and dg2 are identical or proportional."""

  def app(dg):
    lrf = dg(lhs, rhs)
    lr, backprop = jax.vjp(dg, lhs, rhs)
    test_eq("lrf", lrf, lr)
    gl, gr = backprop(gra)
    return lr, gl, gr

  lr1, gl1, gr1 = app(dg1)
  lr2, gl2, gr2 = app(dg2)
  test_eq("lr", lr1 * lr_mult, lr2)  # forward pass
  test_eq("gl", gl1 * gl_mult, gl2)  # backward pass
  test_eq("gr", gr1 * gr_mult, gr2)  # backward pass


class AqtDotGeneralResearchTest(parameterized.TestCase):

  def test_empty(self):
    t = np.random.normal(size=6).reshape((2, 3))
    np.testing.assert_array_equal(t, t)

  def test_fq_noise(self):
    for preserve_zero in [True, False]:
      for prec in [1, 2, 4, 8]:
        for v in [0.1, 1000.0]:
          for seed in range(10):
            key = jax.random.PRNGKey(seed)
            cfg = aqtr.TensorConfig.make(prec)
            cfg.preserve_zero = preserve_zero
            cfg.calib_shared_axes = (0,)
            sample_size = 10000
            shape = (sample_size,)
            a = jax.random.uniform(key, shape, minval=-v, maxval=v)
            qa = aqtr.make_fake_quant(cfg)(a)
            bucket_noise = qa - a  #  ~ U(-bucket_size/2, bucket_size/2)
            bucket_count = (2**prec - 1) if preserve_zero else (2**prec)
            bucket_size = (v * 2) / bucket_count
            noise = bucket_noise / bucket_size + 0.5  # ~U(0, 1)
            pvalue = scipy.stats.kstest(noise, "uniform").pvalue
            assert pvalue > 0.01

  @parameterized.parameters([
      dict(bits=1),
  ])
  def test_fake_quant(
      self,
      bits=4,
      maxval=10.0,
      shape=(20, 1),
  ):
    config = aqtr.TensorConfig.make(bits)
    config.po2_scale = True
    config.calib_shared_axes = (0,)
    x = jnp.linspace(-maxval, maxval, num=shape[0]).reshape(shape)
    grad = jnp.ones(shape) * 12345.0
    x_fq, backprop = jax.vjp(aqtr.make_fake_quant(config), x)
    gx_fq = backprop(grad)
    # print(f"x     =\n{x}")
    # print(f"x_fq  =\n{x_fq}")
    # print(f"grad  =\n{grad}")
    # print(f"gx_fq =\n{gx_fq}")
    del x
    del x_fq
    del grad
    del gx_fq

    # TODO(lew): test

  @parameterized.parameters([
      dict(lhs_bits=1, rhs_bits=1),
      dict(
          lhs_bits=1,
          rhs_bits=1,
          lhs_shape=(3, 2),
          rhs_shape=(2, 4),
          gra_shape=(3, 4),
      ),
      dict(lhs_bits=1, rhs_bits=2),
      dict(lhs_bits=2, rhs_bits=1),
      dict(lhs_bits=2, rhs_bits=2),
      dict(lhs_bits=8, rhs_bits=8),
      dict(lhs_bits=None, rhs_bits=8),
      dict(lhs_bits=8, rhs_bits=None),
      dict(lhs_bits=None, rhs_bits=None),
      dict(
          dims=(((0, 2), (1, 0)), ((3, 1), (2, 4))),
          # contraction: 2, 5; batch: 4, 3
          lhs_shape=(2, 3, 5, 4),  # non-contr: 3, 4
          rhs_shape=(5, 2, 4, 6, 3),  # non-contr: 4, 6, 3
          gra_shape=(4, 3, 6),
      ),
  ])
  def test_dot_general(
      self,
      lhs_bits=1,
      rhs_bits=4,
      lhs_maxval=10.0,
      rhs_maxval=20.0,
      gra_maxval=30.0,
      dims=(((1,), (0,)), ((), ())),  # classical matmul
      # lhs_shape=(1, 2),
      # rhs_shape=(2, 3),
      # gra_shape=(1, 3),  # has to be the shape of the output
      lhs_shape=(10, 20),
      rhs_shape=(20, 30),
      gra_shape=(10, 30),  # has to be the shape of the output
  ):
    # This test is ensuring that `fq_dot_general` and `aqp_dot_general`
    # have the same numerics when scales are power of two (po2).
    # We are passing dims to config so that we can reuse it in fake_quant.
    raw_config = aqtr.DotGeneralRawConfig.make(lhs_bits, rhs_bits)
    # Power-of-2 scales allow FQ and AQT to be exactly the same.
    raw_config.lhs.po2_scale = True
    # Needed if we want to reuse config in the fake_quant.
    raw_config.lhs.calib_shared_axes = dims[0][0]
    raw_config.rhs.po2_scale = True
    raw_config.rhs.calib_shared_axes = dims[0][1]

    config = aqtr.DotGeneralConfig.make()
    config.fwd = raw_config

    # test dot_general
    lhs = rand_unif(lhs_shape, lhs_maxval)
    rhs = rand_unif(rhs_shape, rhs_maxval)
    gra = rand_unif(gra_shape, gra_maxval)

    def aqt_dg_full():
      dg = aqtr.make_dot_general(config)
      return lambda lhs, rhs: dg(lhs, rhs, dims)

    def aqt_dg(use_fake_quant):
      dg = aqtr._make_dot_general_raw(raw_config, use_fake_quant)
      return lambda lhs, rhs: dg(lhs, rhs, dims)[0]

    # Test that with backprop correctly composes 3 functions.
    # We need to test shape calculations and the returned values.
    # For the former we have multiple shapes,
    # for the latter we add some constant and test it on return.
    def lax_dg(lhs, rhs):
      return jax.lax.dot_general(lhs, rhs, dims)

    def lax_dg_248(lhs, rhs):
      def dg_mul(delta):

        def dg(
            lhs,
            rhs,
            dimension_numbers,
            precision=None,
            preferred_element_type=None,
            key=None,
        ):
          ret = jax.lax.dot_general(
              lhs, rhs, dimension_numbers, precision, preferred_element_type
          )
          ret *= delta

          def res(v):
            return aqtr.TensorRes(value=v, qvalue=v, qvalue_scale=1.0)

          res = aqtr.DotGeneralRes(key_bwd=key, lhs=res(lhs), rhs=res(rhs))
          return ret, res

        return dg

      m1 = dg_mul(2.0)
      m2 = dg_mul(4.0)
      m3 = dg_mul(8.0)
      return aqtr._dot_general_raw_attach_gradient(m1, m2, m3)(lhs, rhs, dims)

    check_eq(aqt_dg(False), aqt_dg(True), lhs, rhs, gra)
    check_eq(aqt_dg(False), aqt_dg_full(), lhs, rhs, gra)
    check_eq(
        lax_dg, lax_dg_248, lhs, rhs, gra, lr_mult=2.0, gl_mult=4.0, gr_mult=8.0
    )

  def test_hardware_int8(self):
    def dg(lhs, rhs):
      config = aqtr.DotGeneralRawConfig.make(8, 8)
      config.in_dtype = jnp.int8
      config.out_dtype = jnp.int32
      ret, _ = aqtr._make_dot_general_raw(config)(
          lhs, rhs, dimension_numbers=(((1,), (0,)), ((), ()))
      )
      return ret

    lhs = rand_unif((10, 20), 1.0)
    rhs = rand_unif((20, 30), 1.0)
    [(lhs, rhs, out)] = get_dot_generals(dg, lhs, rhs)
    assert lhs.dtype == jnp.int8
    assert rhs.dtype == jnp.int8
    assert out.dtype == jnp.int32

  @parameterized.parameters([
      (1, 1),
      (1, 2),
      (2, 1),
      (2, 2),
      (8, 8),
      (None, 8),
      (8, None),
      (None, None),
  ])
  def test_conv_general_dilated(
      self,
      lhs_bits,
      rhs_bits,
      lhs_maxval=10.0,
      rhs_maxval=20.0,
  ):
    config = aqtr.make_config_conv_general_dilated(2, lhs_bits, rhs_bits)

    if config.lhs:
      # Power-of-2 scales allow FQ and AQT to be exactly the same.
      config.lhs.po2_scale = True
    if config.rhs:
      config.rhs.po2_scale = True

    batch_n = 10
    contr_n = 20
    feature_n = 30
    lhs = rand_unif((batch_n, 4, 5, contr_n), lhs_maxval)
    rhs = rand_unif((3, 3, contr_n, feature_n), rhs_maxval)

    lax_conv = jax.lax.conv_general_dilated
    aqt_conv = aqtr.make_conv_general_dilated(config)
    kwargs = {
        "window_strides": (1, 1),
        "padding": "SAME",
        "dimension_numbers": fl._conv_dimension_numbers(lhs.shape),
    }
    lhs_fq = aqtr.make_fake_quant(config.lhs)(lhs)
    rhs_fq = aqtr.make_fake_quant(config.rhs)(rhs)
    prod_fq = lax_conv(lhs_fq, rhs_fq, **kwargs)
    prod_aqt = aqt_conv(lhs, rhs, **kwargs)
    assert (prod_aqt == prod_fq).all()


if __name__ == "__main__":
  absltest.main()
