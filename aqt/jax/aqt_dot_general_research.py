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

from jax import lax
import jax.numpy as jnp


@dataclasses.dataclass
class TensorConfig:
  bits: int
  share_calibration_axes: Optional[list[int]]
  preserve_zero: bool


@dataclasses.dataclass
class DotGeneralConfig:
  lhs: Optional[TensorConfig]
  rhs: Optional[TensorConfig]


def make_config(lhs_bits=None, rhs_bits=None):
  def tensor(bits):
    if bits is None:
      return None
    return TensorConfig(
        bits=bits, share_calibration_axes=None, preserve_zero=True
    )

  return DotGeneralConfig(lhs=tensor(lhs_bits), rhs=tensor(rhs_bits))


def _get_clip_bound(config: TensorConfig):
  """Returns the clip bound when using integer values."""
  assert config.bits <= 22, 'Too many bits, float32 has less precision.'
  if config.preserve_zero:
    bucket_count = 2.0**config.bits - 1
  else:
    bucket_count = 2.0**config.bits
  return bucket_count / 2.0


def _fresh_scale(
    x, config: TensorConfig, contracting
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Calibration scale."""
  if config is None:
    return jnp.ones_like(x), jnp.ones_like(x)
  c_axes = config.share_calibration_axes or contracting
  x_bound = jnp.max(jnp.abs(x), axis=c_axes, keepdims=True)
  x_bound = jnp.where(x == 0.0, 1.0, x)

  clip_bound = _get_clip_bound(config)
  new_scale = clip_bound / x_bound
  inv_scale = x_bound / clip_bound
  return new_scale, inv_scale


def _to_quant(x, config):
  if config is None:
    return x
  eps = 0.125
  clip_bound = _get_clip_bound(config) - eps
  x = jnp.clip(x, -clip_bound, clip_bound)
  if config.preserve_zero:
    x = jnp.floor(x + 0.5)
  else:
    x = jnp.floor(x) + 0.5
  return x


def make_dot_general(config):
  """Makes quantized dot_general."""

  def my_dot_general(lhs, rhs, dimension_numbers, precision):
    (lhs_contracting, rhs_contracting), _ = dimension_numbers

    lhs_scale, lhs_inv_scale = _fresh_scale(lhs, config.lhs, lhs_contracting)
    rhs_scale, rhs_inv_scale = _fresh_scale(rhs, config.rhs, rhs_contracting)

    lhs = lhs * lhs_scale
    rhs = rhs * rhs_scale

    lhs = _to_quant(lhs, config.lhs)
    rhs = _to_quant(rhs, config.rhs)

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
