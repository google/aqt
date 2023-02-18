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
rngkey = jax.random.PRNGKey(seed)


def fake_quant(x, prec):
  x_bound = jnp.max(jnp.abs(x))
  scale = (2 ** (prec - 1) - 0.5) / x_bound
  rc = primitives.round_and_clip_to_signed_int
  return rc(x * scale, prec=prec, dtype=x.dtype, half_shift=False) / scale


def rand_unif(shape, maxval):
  global rngkey
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
      # print(new_shape)
      new_shape = (1,) * len(x.shape)
    else:
      new_shape[axis] = 1
    ret2 = jnp.full(new_shape, stddev_of_uniform(maxval / bucket_count))
    # print(ret.shape, ret2.shape)
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
  qa_aqt_fq = aqtr.make_fake_quant(aqtr.make_tensor_config(prec, (0,)))(a)

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
      (1, 1),
      (1, 2),
      (2, 1),
      (2, 2),
      (8, 8),
      (None, 8),
      (8, None),
      (None, None),
  ])
  def test_po2_fake_quant_is_equivalent_with_aqt_dot_general(
      self,
      lhs_bits,
      rhs_bits,
      lhs_maxval=10.0,
      rhs_maxval=20.0,
  ):
    # We are passing dims to config so that we can reuse it in fake_quant.
    config = aqtr.make_dot_general_config(lhs_bits, rhs_bits)

    if config.lhs:
      # Power-of-2 scales allow FQ and AQT to be exactly the same.
      config.lhs.po2_scale = True
      # Needed if we want to reuse config in the fake_quant.
      config.lhs.share_calibration_axes = (1,)
    if config.rhs:
      config.rhs.po2_scale = True
      config.rhs.share_calibration_axes = (0,)

    # test dot_general
    batch_n = 10
    contr_n = 20
    feature_n = 30
    lhs = rand_unif((batch_n, contr_n), lhs_maxval)
    rhs = rand_unif((contr_n, feature_n), rhs_maxval)

    lhs_fq = aqtr.make_fake_quant(config.lhs)(lhs)
    rhs_fq = aqtr.make_fake_quant(config.rhs)(rhs)

    dims = ((1,), (0,)), ((), ())  # classical matmul
    prod_fq = jax.lax.dot_general(lhs_fq, rhs_fq, dims)
    prod_aqt = aqtr.make_dot_general(config)(lhs, rhs, dims)
    assert (prod_aqt == prod_fq).all()

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
  def test_po2_fake_quant_is_equivalent_with_aqt_conv_general_dilated(
      self,
      lhs_bits,
      rhs_bits,
      lhs_maxval=10.0,
      rhs_maxval=20.0,
  ):
    config = aqtr.make_dot_general_config(lhs_bits, rhs_bits)

    if config.lhs:
      # Power-of-2 scales allow FQ and AQT to be exactly the same.
      config.lhs.po2_scale = True
      # Needed if we want to reuse config in the fake_quant.
      config.lhs.share_calibration_axes = [1, 2, 3]
    if config.rhs:
      config.rhs.po2_scale = True
      config.rhs.share_calibration_axes = [0, 1, 2]

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
