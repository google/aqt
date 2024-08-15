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

from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax.v2 import aqt_quantizer
import aqt.jax.v2.aqt_conv_general as aqt_conv
import flax.linen.linear as fl
import jax
import jax.numpy as jnp


def rand_unif(shape, maxval, seed, dtype=jnp.float32):
  key = jax.random.PRNGKey(seed)
  return jax.random.uniform(
      key=key, shape=shape, minval=-maxval, maxval=maxval, dtype=dtype
  )


class AqtConvGeneralTest(parameterized.TestCase):

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
      seed=0,
  ):
    dg_raw_conv = aqt_conv.conv_general_dilated_make(2, lhs_bits, rhs_bits)

    if dg_raw_conv.lhs:
      # Power-of-2 scales allow FQ and AQT to be exactly the same.
      dg_raw_conv.dg_quantizer.lhs.po2_scale = True
    if dg_raw_conv.rhs:
      dg_raw_conv.dg_quantizer.rhs.po2_scale = True

    batch_n = 10
    contr_n = 20
    feature_n = 30
    lhs = rand_unif((batch_n, 4, 5, contr_n), lhs_maxval, seed)
    rhs = rand_unif((3, 3, contr_n, feature_n), rhs_maxval, seed + 1)

    lax_conv = jax.lax.conv_general_dilated
    aqt_conv_fn = aqt_conv.make_conv_general_dilated(dg_raw_conv)
    kwargs = {
        "window_strides": (1, 1),
        "padding": "SAME",
        "dimension_numbers": fl._conv_dimension_numbers(lhs.shape),
    }
    lhs_fq = aqt_quantizer.make_fake_quant(dg_raw_conv.dg_quantizer.lhs)(lhs)
    rhs_fq = aqt_quantizer.make_fake_quant(dg_raw_conv.dg_quantizer.rhs)(rhs)
    prod_fq = lax_conv(lhs_fq, rhs_fq, **kwargs)
    prod_aqt, _ = aqt_conv_fn(lhs, rhs, lhs_qt=None, rhs_qt=None, **kwargs)
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
  def test_conv_general_dilated_quantized(
      self,
      lhs_bits,
      rhs_bits,
      lhs_maxval=10.0,
      rhs_maxval=20.0,
      seed=0,
  ):
    """Check that passing quantized lhs/rhs to aqt_conv_fn works."""
    dg_raw_conv = aqt_conv.conv_general_dilated_make(2, lhs_bits, rhs_bits)
    if dg_raw_conv.lhs:
      # Power-of-2 scales allow FQ and AQT to be exactly the same.
      dg_raw_conv.dg_quantizer.lhs.po2_scale = True
    if dg_raw_conv.rhs:
      dg_raw_conv.dg_quantizer.rhs.po2_scale = True

    batch_n = 10
    contr_n = 20
    feature_n = 30
    lhs = rand_unif((batch_n, 4, 5, contr_n), lhs_maxval, seed)
    lhs_zeros = jnp.zeros(lhs.shape, dtype=lhs.dtype)
    rhs = rand_unif((3, 3, contr_n, feature_n), rhs_maxval, seed + 1)
    rhs_zeros = jnp.zeros(rhs.shape, dtype=rhs.dtype)

    aqt_conv_fn = aqt_conv.make_conv_general_dilated(dg_raw_conv)
    kwargs = {
        "window_strides": (1, 1),
        "padding": "SAME",
        "dimension_numbers": fl._conv_dimension_numbers(lhs.shape),
    }

    lhs_q, _ = dg_raw_conv.dg_quantizer.lhs.quant(
        lhs, calibration_axes=[0, 1, 2, 3])
    rhs_q, _ = dg_raw_conv.dg_quantizer.rhs.quant(
        rhs, calibration_axes=[0, 1, 2])

    out_no_quant, _ = aqt_conv_fn(
        lhs, rhs, lhs_qt=None, rhs_qt=None, **kwargs)

    out_lhs_quant, _ = aqt_conv_fn(
        lhs_zeros, rhs, lhs_qt=lhs_q, rhs_qt=None, **kwargs)
    out_rhs_quant, _ = aqt_conv_fn(
        lhs, rhs_zeros, lhs_qt=None, rhs_qt=rhs_q, **kwargs)
    out_both_quant, _ = aqt_conv_fn(
        lhs_zeros, rhs_zeros, lhs_qt=lhs_q, rhs_qt=rhs_q, **kwargs
    )
    assert (out_no_quant == out_lhs_quant).all()
    assert (out_no_quant == out_rhs_quant).all()
    assert (out_no_quant == out_both_quant).all()


if __name__ == "__main__":
  absltest.main()
