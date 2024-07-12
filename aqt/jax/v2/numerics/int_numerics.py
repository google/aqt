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
"""Numerics for int8, int4, binary and other integer types."""

from typing import Any
from aqt.jax.v2 import stochastic_rounding
from aqt.jax.v2 import utils
from aqt.jax.v2.numerics import numerics
from jax import lax
import jax.numpy as jnp


def _apply_noise_fn(x, noise_fn, context):
  assert context.key is not None, (
      'noise_fn is set, requesting stochastic rounding, but RNG was not '
      'passed in Context.key'
  )
  return (x + noise_fn(x.shape, context.key)).astype(x.dtype)


def _apply_gradient_of_clip(res, grad, lower_clip_bound, upper_clip_bound):
  # For boundary values we will have the full gradient.
  # When using max(abs(x)) scaling, x is always in the interior, and the
  # gradient clip is always 1. So, we can always set clip_gradient to false.
  # However, other types of scaling may result in x being outside (i.e., there
  # is clipping). In that case it may be desirable to make the gradient zero.
  (x,) = res
  return grad * (lower_clip_bound <= x) * (x <= upper_clip_bound)


@utils.flax_slots_kw_only_dataclass
class IntNumerics(numerics.AqtNumerics):
  """Abstract class for integer numerics typing."""


@utils.flax_slots_kw_only_dataclass
class IntSymmetric(IntNumerics):
  """Symmetric numerics for sint8, sint4, binary, etc."""

  bits: int
  preserve_zero: bool
  # false = map max val on the end of the last bucket
  # true = map max val on the middle of the last
  preserve_max_val: bool
  # The quantized values are only guaranteed to be within the appropriate signed
  # int range if clip=True and round=True. Otherwise, the values are only
  # guaranteed to be within [sint_min, sint_max + 1]. The range may be more
  # restricted depending on the full configuration.
  clip: bool
  clip_gradient: bool
  round: bool
  noise_fn: None | stochastic_rounding.NoiseFn
  dtype: None | Any = None

  # pylint: disable=line-too-long
  # Verifying the correctness of these functions amounts to verifying this table:
  # if preserve_zero == F, zero might be rounded either to [-1, 0] bucket or to [0, 1] bucket
  # preserve_zero, preserve_max_val, 8b, 2b, 1b
  # F, F, 128.0, 2.0, 1.0  # bucket count is even; map onto the far edge of the last bucket
  # F, T, 127.5, 1.5, 0.5  # bucket count is even; map onto the center of the last bucket
  # T, F, 127.5, 1.5, 0.5  # bucket count is odd;  map onto the far edge of the last bucket
  # T, T, 127.0, 1.0, 0.0  # bucket count is odd;  map onto the center of the last bucket
  # pylint: enable=line-too-long

  def get_edge_of_last_int_bucket(self):
    ret = 2.0 ** (self.bits - 1)
    if self.preserve_zero:
      # Lose one bucket.
      ret -= 0.5
    return ret

  def get_center_of_last_int_bucket(self):
    return self.get_edge_of_last_int_bucket() - 0.5

  def get_quant_bound(self):
    if self.preserve_max_val:
      return self.get_center_of_last_int_bucket()
    else:
      return self.get_edge_of_last_int_bucket()

  def _get_fwd_clip_bound(self):
    # If we are not rounding, we just clip to bucket edges.
    fwd_clip_bound = self.get_edge_of_last_int_bucket()
    # If, after clip, we are rounding, we need to make sure that
    # we won't round values at the clip_bound away to the
    # non-existing bucket.
    if self.round:
      # Reducing fwd_clip_bound by any value in (0.0, 1.0) is correct.
      fwd_clip_bound -= 0.5
    return fwd_clip_bound

  def get_dtype(self):
    return self.dtype

  def vjp_fwd(self, x, context):
    """Forward pass."""
    res = (x,)
    return_dtype = self.dtype if self.dtype is not None else x.dtype
    assert self.bits <= 22, 'Too many bits, float32 has less precision.'

    if self.noise_fn:
      x = _apply_noise_fn(x, self.noise_fn, context)

    if self.clip:
      fwd_clip_bound = self._get_fwd_clip_bound()
      x = jnp.clip(x, -fwd_clip_bound, fwd_clip_bound)

    # Maybe round
    if self.round:
      # TODO(lew): Have bucket centers at 2*k + 1, not at halves.
      round_to_halves = not self.preserve_zero
      if round_to_halves:
        x = jnp.floor(x) + 0.5
      else:
        x = lax.round(x, lax.RoundingMethod.TO_NEAREST_EVEN)

    x = x.astype(return_dtype)  # cast to user-specified dtype or input dtype.
    return x, res

  def vjp_bwd(self, res, grad):
    ret = grad
    if self.clip_gradient:
      clip_bound = self._get_fwd_clip_bound()
      ret = _apply_gradient_of_clip(res, grad, -clip_bound, clip_bound)
    return (ret, None)


@utils.flax_slots_kw_only_dataclass
class IntAsymmetric(IntNumerics):
  """Asymmetric numerics for sint8, sint4, binary, etc."""

  bits: int
  clip: bool
  clip_gradient: bool
  round: bool
  noise_fn: None | stochastic_rounding.NoiseFn
  dtype: None | Any = None

  def get_dtype(self):
    return self.dtype

  def get_quant_bound(self):
    return 2.0**self.bits - 1

  def get_quant_range(self):
    if self.bits > 1:
      # Full signed int range.
      sint_max = 2.0 ** (self.bits - 1) - 1
      sint_min = -(2.0 ** (self.bits - 1))
      return sint_min, sint_max
    else:
      # Boolean range.
      return 0.0, 1.0

  def vjp_fwd(self, x, context):
    """Forward pass."""
    res = (x,)
    return_dtype = self.dtype if self.dtype is not None else x.dtype
    assert self.bits <= 22, 'Too many bits, float32 has less precision.'

    if self.noise_fn:
      x = _apply_noise_fn(x, self.noise_fn, context)

    if self.clip:
      x = jnp.clip(x, *self.get_quant_range())

    if self.round:
      x = lax.round(x, lax.RoundingMethod.TO_NEAREST_EVEN)

    x = x.astype(return_dtype)  # cast to user-specified dtype or input dtype.
    return x, res

  def vjp_bwd(self, res, grad):
    ret = grad
    if self.clip_gradient:
      ret = _apply_gradient_of_clip(res, grad, *self.get_quant_range())
    return (ret, None)
