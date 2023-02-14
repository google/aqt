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

import dataclasses
from typing import Optional, Tuple

import jax
from jax import lax
import jax.numpy as jnp


@dataclasses.dataclass
class TensorConfig:
  bits: int
  share_calibration_axes: Optional[list[int]]
  preserve_zero: bool
  bound: Optional[float]
  bound_stop_grad: bool
  # false = map max val on the end of the last bucket
  # true = map max val on the middle of the last
  preserve_max_val: bool


def make_tensor_config(bits, share_calibration_axes) -> TensorConfig | None:
  if bits is None:
    return None
  return TensorConfig(
      bits=bits,
      share_calibration_axes=share_calibration_axes,
      preserve_zero=False if bits == 1 else True,
      bound=None,
      bound_stop_grad=True,
      preserve_max_val=False,
  )


@dataclasses.dataclass
class DotGeneralConfig:
  lhs: Optional[TensorConfig]
  rhs: Optional[TensorConfig]


def make_config(lhs_bits=None, rhs_bits=None) -> DotGeneralConfig:
  """Create quantization configs for input matrices to a matmul."""
  return DotGeneralConfig(
      lhs=make_tensor_config(lhs_bits, None),
      rhs=make_tensor_config(rhs_bits, None),
  )

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


def _fresh_scale(
    x, config: TensorConfig, contracting: list[int] | None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Calibration scale."""
  if config is None:
    scale_one = jnp.ones([1 for _ in x.shape], dtype=x.dtype)
    return scale_one, scale_one
  # We have 2 sources for contraction axes:
  c_axes = config.share_calibration_axes
  a_axes = contracting
  msg = 'We need contraction axes either in config or as an argument.'
  assert c_axes or a_axes, msg
  if c_axes is not None and a_axes is not None:
    assert c_axes == a_axes, (c_axes, a_axes)

  # x_bound is the input range that gets mapped to the integer clip_bound
  # For dynamic quant x_bound = max(x); for static quant x_bound = config.bound
  if config.bound is None:
    x_bound = jnp.max(jnp.abs(x), axis=c_axes or a_axes, keepdims=True)
  else:
    assert config.bound > 0, 'Static quantization bound should be positive.'
    x_bound = jnp.asarray(config.bound)
  x_bound = jnp.where(x_bound == 0.0, 1.0, x_bound)
  if config.bound_stop_grad:
    x_bound = lax.stop_gradient(x_bound)

  # This is the value that the x_bound is mapped to.
  x_bound_repr = _get_max_value_representation(config)
  new_scale = x_bound_repr / x_bound
  inv_scale = x_bound / x_bound_repr
  return new_scale, inv_scale


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
  clip_bound = _get_clip_bound(config)
  x = jnp.clip(x, -clip_bound, clip_bound)
  return _round(x, round_to_halves=not config.preserve_zero)


def make_fake_quant(config: Optional[TensorConfig]):
  def fake_quant(x):
    scale, inv_scale = _fresh_scale(x, config, None)
    x = x * scale
    x = _clip_and_round(x, config)
    x = x * inv_scale
    return x

  return fake_quant


def make_dot_general(config: Optional[DotGeneralConfig]):
  """Makes quantized dot_general."""
  if config is None:
    config = DotGeneralConfig(None, None)

  def my_dot_general(lhs, rhs, dimension_numbers, precision=None):
    (lhs_contracting, rhs_contracting), _ = dimension_numbers

    lhs_scale, lhs_inv_scale = _fresh_scale(lhs, config.lhs, lhs_contracting)
    rhs_scale, rhs_inv_scale = _fresh_scale(rhs, config.rhs, rhs_contracting)
    lhs = lhs * lhs_scale
    rhs = rhs * rhs_scale

    lhs = _clip_and_round(lhs, config.lhs)
    rhs = _clip_and_round(rhs, config.rhs)

    ret = lax.dot_general(
        lhs, rhs, dimension_numbers=dimension_numbers, precision=precision
    )

    combined_inv_scale = lax.dot_general(
        lhs_inv_scale,
        rhs_inv_scale,
        dimension_numbers=dimension_numbers,
        precision=precision,
    )
    ret = ret * combined_inv_scale
    return ret

  return my_dot_general
