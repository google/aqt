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
from aqt.jax_legacy.jax import primitives

import flax.linen.linear as fl
import jax
import jax.numpy as jnp
import numpy as np


# seed = np.random.randint(0, 100)
seed = 0
rngkey = None


def get_dot_generals(f, *args):
  jaxpr = jax.make_jaxpr(f)(*args)
  for i in jaxpr.eqns:
    if i.primitive.name == "dot_general":
      [lhs, rhs] = i.invars
      [out] = i.outvars
      yield (lhs.aval, rhs.aval, out.aval)


def fake_quant(x, prec):
  x_bound = jnp.max(jnp.abs(x))
  scale = (2 ** (prec - 1) - 0.5) / x_bound
  rc = primitives.round_and_clip_to_signed_int
  return rc(x * scale, prec=prec, dtype=x.dtype, half_shift=False) / scale


def rand_unif(shape, maxval):
  global rngkey
  if rngkey is None:
    rngkey = jax.random.PRNGKey(seed)
  k1, k2 = jax.random.split(rngkey)
  rngkey = k1
  return jax.random.uniform(key=k2, shape=shape, minval=-maxval, maxval=maxval)


def stddev_of_uniform(maxval):
  return jnp.sqrt((2 * maxval) ** 2 / 12.0)


def bucket_abs_noise(x, *, prec, axis, preserve_zero=True):
  """noise_in_bucket * bucket_size ."""
  # TODO(lew): use preserve_zero
  # 2* becasue interval is [-maxabs(x), maxabs(x)]
  interval_length = 2 * jnp.max(jnp.abs(x), axis=axis, keepdims=True)
  bucket_count = 2**prec
  if preserve_zero:
    bucket_count -= 1
  # We assuming here presever_max_val = false because we assume usage of the
  # whole length of the edge buckets.
  # presever_max_val = true would mean we use only half of edge buckets.
  bucket_size = interval_length / bucket_count
  noise_in_bucket = 1.0 / 4.0
  return bucket_size * noise_in_bucket


def expected_std_noise(x, *, prec, axis):
  interval_length = 2 * jnp.max(jnp.abs(x), axis=axis, keepdims=True)
  bucket_count = 2**prec - 1
  bucket_size = interval_length / bucket_count
  ret = bucket_size / jnp.sqrt(12.0)

  def test():
    maxval = jnp.max(jnp.abs(x), axis=axis, keepdims=True)
    new_shape = list(x.shape)
    if axis is None:
      new_shape = (1,) * len(x.shape)
    else:
      new_shape[axis] = 1
    ret2 = jnp.full(new_shape, stddev_of_uniform(maxval / bucket_count))
    assert (jnp.abs(ret - ret2) < 0.001).all(), (ret, ret2)

  test()
  return ret


def fq_noise_test(*, prec, maxval, sample_size, do_print=False):
  a = rand_unif((sample_size,), maxval)
  # aqt_config = aqtr.make_config(prec, prec)
  # aqt_config = None  # TODO(lew): make a separate unit test
  # dot_general = aqtr.make_dot_general(aqt_config)
  # axes = (((0,), (0,)), ((), ()))
  # qa_aqt_dg = dot_general(a, np.identity(sample_size), axes)

  qa_fq = fake_quant(a, prec)
  cfg = aqtr.make_tensor_config(prec, (0,))
  cfg.calib_shared_axes = (0,)
  qa_aqt_fq = aqtr.make_fake_quant(cfg)(a)

  def compare(name, qa):
    mean_abs_noise = jnp.mean(jnp.abs((qa - a)))
    mean_std_noise = jnp.sqrt(jnp.mean((qa - a) ** 2))

    # We need to show that mean_abs_noise is close to bucket_abs_noise
    rel_abs_noise = jnp.log(
        mean_abs_noise / bucket_abs_noise(a, prec=prec, axis=None)
    )
    rel_std_noise = jnp.log(
        mean_std_noise / expected_std_noise(a, prec=prec, axis=None)
    )
    rel_abs_noise_scaled = rel_abs_noise * jnp.sqrt(sample_size)
    rel_std_noise_scaled = rel_std_noise * jnp.sqrt(sample_size)

    if do_print:
      # print(f"max_a={max_a}")
      # print(f"sa={sa}")
      print(name)
      print(qa.shape, a.shape)
      if qa.size < 10:
        print("a", a)
        print("qa", qa)
      # print(f"bucket_abs_noise={bucket_abs_noise}")
      print(f"mean_abs_noise={mean_abs_noise}")
      # print(f"noise_test={noise_test}")
      print(
          "rel_abs_noise *"
          f" jnp.sqrt(sample_size)={rel_abs_noise * jnp.sqrt(sample_size)}"
      )
      print(f"mean_std_noise={mean_std_noise}")
      print(
          "expected_std_noise(a, prec=prec,"
          f" axis=None)={expected_std_noise(a, prec=prec, axis=None)}"
      )
      print(f"rel_std_noise={rel_std_noise}")
      print(
          f"sample_size={sample_size}; {rel_abs_noise_scaled};"
          f" {rel_std_noise_scaled}"
      )
      print()

    # TODO(lew): This bound is too lax. We need to investigate or improve.
    assert -4.0 < rel_abs_noise < 4.0  # standard sufficient test
    assert -4.0 < rel_std_noise < 4.0  # standard sufficient test
    # TODO(lew): Why did I have to comment out these tests?
    # more sensitive tests:
    # assert -4.0 < rel_abs_noise_scaled < 4.0, rel_abs_noise_scaled
    # assert -4.0 < rel_std_noise_scaled < 4.0, rel_std_noise_scaled

  compare("fake_quant", qa_fq)
  compare("aqt fake_quant", qa_aqt_fq)
  # compare("dot_general", qa_aqt_dg)


# The main test strategyis :
# - Test that FQ is sensible (improve fq_noise_test)
# - Test that we really use int8/int4/bool in XLA/HLO
# - Test of fq vs dot_general equivalence in po2
# - Test that it all applies to gradients
# TODO(lew): Tests are a bit incomplete. What we need:
# On top of that we shell test:
#  - Gradinent is 1 in the range, 0 outside the abs-max range.


def test_eq(name, a, b):
  # TODO(lew): use library function.
  mean_err = jnp.mean(jnp.abs(a - b))
  if mean_err != 0.0:
    print(name)
    print(mean_err)
    print(a.shape)
    print(a[:3, :3])
    print(b.shape)
    print(b[:3, :3])
    assert False


class AqtDotGeneralResearchTest(parameterized.TestCase):

  def test_empty(self):
    t = np.random.normal(size=6).reshape((2, 3))
    np.testing.assert_array_equal(t, t)

  def test_noise_levels(self):
    for prec in [1, 2, 4, 8]:
      for maxval in [0.1, 1000.0]:
        for sample_size in [1, 2, 100]:
          for _ in range(20):
            fq_noise_test(
                prec=prec,
                maxval=maxval,
                sample_size=sample_size,
            )

  @parameterized.parameters([
      dict(bits=1),
  ])
  def test_fake_quant(
      self,
      bits=4,
      maxval=10.0,
      shape=(20, 1),
  ):
    config = aqtr.make_tensor_config(bits)
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
      lhs_shape=(10, 20),
      rhs_shape=(20, 30),
      gra_shape=(10, 30),  # has to be the shape of the output
  ):
    # This test is ensuring that `fq_dot_general` and `aqp_dot_general`
    # have the same numerics when scales are power of two (po2).
    # We are passing dims to config so that we can reuse it in fake_quant.
    config = aqtr.make_dot_general_config(lhs_bits, rhs_bits)
    if config.lhs:
      # Power-of-2 scales allow FQ and AQT to be exactly the same.
      config.lhs.po2_scale = True
      # Needed if we want to reuse config in the fake_quant.
      config.lhs.calib_shared_axes = dims[0][0]
    if config.rhs:
      config.rhs.po2_scale = True
      config.rhs.calib_shared_axes = dims[0][1]

    # test dot_general
    lhs = rand_unif(lhs_shape, lhs_maxval)
    rhs = rand_unif(rhs_shape, rhs_maxval)
    gra = rand_unif(gra_shape, gra_maxval)

    def check_eq(dg1, dg2, lr_mult=1.0, gl_mult=1.0, gr_mult=1.0):
      """Tests whether dg1 and dg2 are identical or proportional."""
      def app(dg):
        lr, backprop = jax.vjp(dg, lhs, rhs)
        gl, gr = backprop(gra)
        return lr, gl, gr

      lr1, gl1, gr1 = app(dg1)
      lr2, gl2, gr2 = app(dg2)
      test_eq("lr", lr1 * lr_mult, lr2)  # forward pass
      test_eq("gl", gl1 * gl_mult, gl2)  # backward pass
      test_eq("gr", gr1 * gr_mult, gr2)  # backward pass

    def aqt_dg(use_fake_quant):
      dg = aqtr.make_dot_general(config, use_fake_quant)
      return lambda lhs, rhs: dg(lhs, rhs, dims)

    check_eq(aqt_dg(False), aqt_dg(True))

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
        ):
          return (
              jax.lax.dot_general(
                  lhs, rhs, dimension_numbers, precision, preferred_element_type
              )
              * delta
          )

        return dg

      m1 = dg_mul(2.0)
      m2 = dg_mul(4.0)
      m3 = dg_mul(8.0)
      return aqtr.dot_general_with_gradient(m1, m2, m3)(lhs, rhs, dims)

    check_eq(lax_dg, lax_dg_248, lr_mult=2.0, gl_mult=4.0, gr_mult=8.0)

  def test_hardware_int8(self):
    def dg(lhs, rhs):
      config = aqtr.make_dot_general_config(8, 8)
      config.use_hardware_int8 = True
      return aqtr.make_dot_general(config)(
          lhs, rhs, dimension_numbers=(((1,), (0,)), ((), ()))
      )

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
