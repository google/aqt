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
"""Quantized dot_general."""

import copy
import dataclasses
from typing import Optional
import jax
from jax import lax
import jax.numpy as jnp


@dataclasses.dataclass
class TensorConfig:
  """Configuration of quantization of one tensor or one side of tensor op."""
  bits: int
  calib_shared_axes: Optional[list[int]]
  preserve_zero: bool
  bound: Optional[float]
  bound_stop_grad: bool
  # false = map max val on the end of the last bucket
  # true = map max val on the middle of the last
  preserve_max_val: bool
  clip: bool
  round: bool
  # Round up the calibration to power of 2 (po2).
  po2_scale: bool


def make_tensor_config(bits) -> TensorConfig | None:
  if bits is None:
    return None
  return TensorConfig(
      bits=bits,
      calib_shared_axes=None,
      preserve_zero=False if bits == 1 else True,
      bound=None,
      bound_stop_grad=True,
      preserve_max_val=False,
      clip=True,
      round=True,
      po2_scale=False,
  )


@dataclasses.dataclass
class DotGeneralConfig:
  lhs: Optional[TensorConfig]
  rhs: Optional[TensorConfig]


def make_dot_general_config(lhs_bits=None, rhs_bits=None) -> DotGeneralConfig:
  """Create quantization configs for input matrices to a matmul."""
  return DotGeneralConfig(
      lhs=make_tensor_config(lhs_bits),
      rhs=make_tensor_config(rhs_bits),
  )


def make_config_conv_general_dilated(
    spatial_dimensions=2,
    lhs_bits=None,
    rhs_bits=None,
) -> DotGeneralConfig:
  config = make_dot_general_config(lhs_bits, rhs_bits)
  # Hardcoding flax assumptions.
  if config.lhs:
    config.lhs.calib_shared_axes = list(range(1, spatial_dimensions + 2))
  if config.rhs:
    config.rhs.calib_shared_axes = list(range(0, spatial_dimensions + 2 - 1))
  return config


# The arithmetic of scaling, clipping and rounding can be confusing.
# Here I add some documentation to hopefully clarify it.
# Bucket is an interval of rounding after the scaling is applied.
# Facts:
#  - Bucket size is always 1.0.
#  - Middle of the bucket = righ of the bucket - 0.5
#  - If not preserve_zero then
#    - bucket ends align with integers.
#    - bucket center align with integers+0.5 .
#  - If preserve_zero then
#    - bucket ends align with intigers+0.5.
#    - bucket center align with integers.
#  - We have two types of rounding, both mostly unbiased:
#    - round_int0(x) = floor(x+0.5) # rounding to integer
#    - round_int5(x) = floor(x)+0.5 # rounding to integer+0.5
#  - Center of the bucket is presereved by rounding in all cases.

# Let's explore 6 options:
# preserve_zero = False - rounding to x.5. i.e 0.5, 1.5, etc are preserved
#   prec=2
#   buckets: [-2, -1] [-1, 0] [0, 1] [1, 2]
#   bucket centers: -1.5 -0.5 0.5 1.5
#     preserve_max_val = False
#     we map largest value to 2.0 (mapping to the end of largest bucket)
#     preserve_max_val = True
#     we map largest value to 1.5
#   prec=1
#   bucket centers: -0.5 0.5
#   buckets: [-1, 0] [0, 1]
#     preserve_max_val = False
#     we map largest value to 1.0
#     preserve_max_val = True
#     we map largest value to 0.5
# preserve_zero = True - rounding to x.0 i.e 0.0, 1.0, 2.0, etc are preserved
#   prec=2
#   buckets: [-1.5, -0.5] [-0.5, 0.5] [0.5, 1.5]
#   bucket centers: -1, 0, 1
#     preserve_max_val = False
#     we map largest value to 1.5 (mapping to the end of largest bucket)
#     preserve_max_val = True
#     we map largest value to 1.0

# Summary in the table.
# preserve_zero, preserve_max_val, max_val_mapped_to, clipping_formula
## True , False , 2^(n-1) - 0.5, round_int0(clip(x, 2^(n-1) - 0.5 - eps))
# True , True  , 2^(n-1) - 1.0, round_int0(clip(x, 2^(n-1) - 0.5 - eps))
# False, False , 2^(n-1)      , round_int5(clip(x, 2^(n-1) - 0.0 - eps))
# False, True  , 2^(n-1) - 0.5, round_int5(clip(x, 2^(n-1) - 0.0 - eps))
#
# Clipping is there only to round all the buckets beyond prec to the biggest
# bucket. (we will have a separate method for gradients)
#
# eps can be anywhere in (0, 1.0) for correctness (sharp inequalities).
# We choose eps=0.5


def _get_max_value_representation(config: TensorConfig):
  """Largest quantization tensor value is mapped onto 'int' value returned by this function."""
  assert config.bits <= 22, 'Too many bits, float32 has less precision.'
  clip_bound = 2.0 ** (config.bits - 1)
  if config.preserve_zero:
    clip_bound -= 0.5
  if config.preserve_max_val:
    clip_bound -= 0.5
  return clip_bound


def _get_clip_bound(config: TensorConfig):
  """Returns the clip bound when using integer values."""
  assert config.bits <= 22, 'Too many bits, float32 has less precision.'
  eps = 0.5
  clip_bound = 2.0 ** (config.bits - 1) - eps
  if config.preserve_zero:
    clip_bound -= 0.5
  return clip_bound


def _fresh_scale(x, config: TensorConfig) -> jnp.ndarray:
  """Calibration scale."""
  if config is None:
    return jnp.ones((1,) * len(x.shape), dtype=x.dtype)

  # We have 2 sources for contraction axes:
  assert config.calib_shared_axes

  # x_bound is the input range that gets mapped to the integer clip_bound
  # For dynamic quant x_bound = max(x); for static quant x_bound = config.bound
  if config.bound is None:
    x_bound = jnp.max(jnp.abs(x), axis=config.calib_shared_axes, keepdims=True)
  else:
    assert config.bound > 0, 'Static quantization bound should be positive.'
    x_bound = jnp.asarray(config.bound)
  x_bound = jnp.where(x_bound == 0.0, 1.0, x_bound)
  if config.bound_stop_grad:
    x_bound = lax.stop_gradient(x_bound)

  # This is the value that the x_bound is mapped to.
  x_bound_repr = _get_max_value_representation(config)
  new_scale = x_bound_repr / x_bound
  if config.po2_scale:
    new_scale = 2 ** jnp.ceil(jnp.log2(new_scale))
  return new_scale


@jax.custom_jvp
def floor_with_ste(x):
  """Floor with Straight-Through-Estimator gradient."""
  return jnp.floor(x)


# Add STE.
floor_with_ste.defjvp(
    lambda primals, tangents: (floor_with_ste(primals[0]), tangents[0])
)


def _round(x, round_to_halves=False):
  """(Mostly) unbiased rounding to either an integer or integer+0.5 ."""
  if round_to_halves:
    return floor_with_ste(x) + 0.5
  else:
    return floor_with_ste(x + 0.5)


def _clip_and_round(x, config: TensorConfig):
  if config is None:
    return x
  if config.clip:
    clip_bound = _get_clip_bound(config)
    x = jnp.clip(x, -clip_bound, clip_bound)
  if config.round:
    x = _round(x, round_to_halves=not config.preserve_zero)
  return x


def make_fake_quant(config: Optional[TensorConfig]):
  def fake_quant(x):
    if not config:
      return x
    scale = _fresh_scale(x, config)
    x = x * scale
    x = _clip_and_round(x, config)
    x = x / scale
    return x

  return fake_quant


def make_dot_general(config: Optional[DotGeneralConfig]):
  """Makes quantized lax.dot_general replacement."""
  config = copy.copy(config)
  if config is None:
    config = DotGeneralConfig(None, None)

  def my_dot_general(lhs, rhs, dimension_numbers, precision=None):
    # Axes can be partitioned into:
    # - contraction axes (ca)
    # - batch axes (ba)
    # - remaining axes (ra).
    (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
    if config.lhs:
      config.lhs.calib_shared_axes = config.lhs.calib_shared_axes or lhs_ca
      lhs_scale = _fresh_scale(lhs, config.lhs)
      lhs = lhs * lhs_scale
      lhs = _clip_and_round(lhs, config.lhs)

    if config.rhs:
      config.rhs.calib_shared_axes = config.rhs.calib_shared_axes or rhs_ca
      rhs_scale = _fresh_scale(rhs, config.rhs)
      rhs = rhs * rhs_scale
      rhs = _clip_and_round(rhs, config.rhs)

    out = lax.dot_general(
        lhs, rhs, dimension_numbers=dimension_numbers, precision=precision
    )

    def scale_trans(x, ca, ba, pre_ones=0, post_ones=0):
      for i in ca:
        assert x.shape[i] == 1
      ra = tuple(i for i in range(len(x.shape)) if i not in ba + ca)
      x = jnp.transpose(x, ba + ra + ca)
      s1 = x.shape[: len(ba)]
      s2 = x.shape[len(ba) : -len(ca)]
      x = x.reshape(s1 + (1,) * pre_ones + s2 + (1,) * post_ones)
      return x

    if config.lhs:
      rhs_ra_n = len(rhs.shape) - len(rhs_ca) - len(rhs_ba)
      lhs_scale_t = scale_trans(lhs_scale, lhs_ca, lhs_ba, post_ones=rhs_ra_n)
      out = out / lhs_scale_t

    if config.rhs:
      lhs_ra = len(lhs.shape) - len(lhs_ca) - len(lhs_ba)
      rhs_scale_t = scale_trans(rhs_scale, rhs_ca, rhs_ba, pre_ones=lhs_ra)
      out = out / rhs_scale_t

    return out

  return my_dot_general


def make_conv_general_dilated(config: Optional[DotGeneralConfig]):
  """Makes quantized lax.make_conv_general_dilated replacement."""
  # TODO(lew): Either rename DotGeneralConfig or make a conv-specific config.
  config = copy.copy(config)
  if config is None:
    config = DotGeneralConfig(None, None)

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

    if config.lhs:
      # Flax assumptions.
      assert config.lhs.calib_shared_axes == list(range(1, rank))
      lhs_scale = _fresh_scale(lhs, config.lhs)
      lhs = lhs * lhs_scale
      lhs = _clip_and_round(lhs, config.lhs)

    if config.rhs:
      assert config.rhs.calib_shared_axes == list(range(0, rank - 1))
      rhs_scale = _fresh_scale(rhs, config.rhs)
      rhs = rhs * rhs_scale
      rhs = _clip_and_round(rhs, config.rhs)

    out = lax.conv_general_dilated(
        lhs=lhs,
        rhs=rhs,
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

    if config.lhs:
      out /= lhs_scale

    if config.rhs:
      out /= rhs_scale
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
