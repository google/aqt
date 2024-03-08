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
"""Quantized conv_general."""

# Lingo in this file:
#
# - lhs(rhs) - left(right) hand side of a binary operation
# - ca - contraction axes
# - ba - batch axes
# - ra - remaining axes

# pylint: disable=protected-access

from aqt.jax.v2 import aqt_tensor
from aqt.jax.v2 import config
import jax
from jax import lax
import jax.numpy as jnp


def make_conv_general_dilated(cfg: config.DotGeneralRaw):
  """Makes quantized lax.make_conv_general_dilated replacement."""
  # TODO(lew): Either rename DotGeneralConfig or make a conv-specific cfg.
  if cfg is None:
    cfg = config.DotGeneralRaw.make()

  def my_conv_general_dilated(
      lhs,
      rhs,
      window_strides,
      padding,
      lhs_dilation=None,
      rhs_dilation=None,
      dimension_numbers=None,
      feature_group_count=1,
      batch_group_count=1,
      precision=None,
      preferred_element_type=None,
  ) -> jax.Array:
    msg1 = """
To simplify the code, we currently assume a Flax-particular layout of the data.
This makes sense, because this is the main use-case of this function.
However if there is any other use, we will drop that assumption."""
    rank = len(lhs.shape)
    assert len(rhs.shape) == rank
    assert dimension_numbers is not None, msg1
    assert dimension_numbers.lhs_spec[0:2] == (0, rank - 1), msg1
    assert dimension_numbers.rhs_spec[0:2] == (rank - 1, rank - 2), msg1
    assert dimension_numbers.out_spec[0:2] == (0, rank - 1), msg1
    # In Flax, lhs is the inputs, rhs is the kernel.
    # lhs layout is B, spatials..., Ci
    # rhs layout is: spatials..., Ci, Co
    # out layous it: B, spatials..., Co
    #
    # we need to share these axes: lhs[1:] , rhs[:-1]
    # we have a scale/invscale per: lhs[0] / out[0] and rhs[-1] / out[-1]

    # Flax assumptions.
    assert cfg.lhs.quantizer.calib_shared_axes == list(range(1, rank))
    assert cfg.rhs.quantizer.calib_shared_axes == list(range(0, rank - 1))

    lhs_qt, _ = cfg.lhs.quantizer.quant(lhs, calibration_axes=None)
    rhs_qt, _ = cfg.rhs.quantizer.quant(rhs, calibration_axes=None)

    # lax.conv_general_dilated does not support int8 * float.
    # Therefore, cast qvalue back to its original data dtype.
    # Delete the following two lines when the constraint is lifted.
    lhs_qt = lhs_qt.qvalue_astype(lhs.dtype)
    rhs_qt = rhs_qt.qvalue_astype(rhs.dtype)

    out = lax.conv_general_dilated(
        lhs=lhs_qt.qvalue,
        rhs=rhs_qt.qvalue,
        window_strides=window_strides,
        padding=padding,
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )

    # It seems lucky that original scale has shape suitable for output
    # scaling without any transposition.
    out = aqt_tensor.QTensor(
        qvalue=out,
        scale=[],
        scale_t=None,
        dequant_dtype=jnp.promote_types(lhs, rhs),
    )
    assert out.scale is not None  # pytype help
    out.scale.extend(lhs_qt.scale)
    out.scale.extend(rhs_qt.scale)
    out = out.dequant()

    # # Future scale granularity optimization.
    # In 1x1 conv, each pixel (spatial location) can have different scales
    # in 1xN (rows x colums) conv each row can have different scale, but
    # columns need to share the scales ,  because we are adding pixels across.
    #
    # For patch convs we could have separate scales per patch.
    # We don't do that optimization, because there is a  Flax op: ConvLocal
    # using lax.conv_general_dilated_local which uses lax.dot_general.
    #
    # Dilations: If a dilation of LHS is bigger than the total spatial size of
    # RHS, we could use separe (per LHS pixel) scales.
    # The same applies to dilated RHS.
    # We don't do that optimization yet.
    #
    # We can have different scales across different groups.
    # This applies to both feature and batch.
    return out

  return my_conv_general_dilated
