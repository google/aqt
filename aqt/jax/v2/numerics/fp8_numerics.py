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
"""Numerics for fp8."""

from typing import Literal, Optional, TypeAlias
from aqt.jax.v2 import stochastic_rounding
from aqt.jax.v2 import utils
from aqt.jax.v2.numerics import numerics
import jax
import jax.numpy as jnp


FP8Dtype: TypeAlias = Literal['e4m3', 'e5m2']
fp8_map = {'e4m3': jnp.float8_e4m3fn, 'e5m2': jnp.float8_e5m2}


@utils.flax_slots_dataclass
class Fp8Numerics(numerics.AqtNumerics):
  """Numerics for fp8."""

  dtype: Literal[jnp.float8_e4m3fn, jnp.float8_e5m2]
  exponent_bits: int = 4
  mantissa_bits: int = 3
  noise_fn: Optional[stochastic_rounding.NoiseFn] = None

  def _get_edge_of_last_fp8_bucket(self):
    return jnp.finfo(self.dtype).max.astype(jnp.bfloat16)

  def get_dtype(self):
    return self.dtype

  def abs_val_mapped_to(self):
    return self._get_edge_of_last_fp8_bucket()

  def vjp_fwd(self, x, context):
    res = (x,)
    if not (
        (self.exponent_bits == 4 and self.mantissa_bits == 3)
        or (self.exponent_bits == 5 and self.mantissa_bits == 2)
    ):
      raise ValueError(
          '(exponent_bits, mantissa_bits) can only be (4,3) or (5,2) but was '
          f'({self.exponent_bits}, {self.mantissa_bits})'
      )

    if self.noise_fn is not None:
      x = (x + self.noise_fn(x.shape, context.key)).astype(x.dtype)

    # clip
    fwd_clip_bound = self._get_edge_of_last_fp8_bucket()
    x = jnp.clip(x, -1 * fwd_clip_bound, fwd_clip_bound)

    # round
    x = round_to_nearest_even(x, self.dtype)

    return x, res

  def vjp_bwd(self, res, grad):
    # This is gradient of clip.
    # For boundary values we will have full gradient.
    # We might use something like this for calibrations other than abs(max(x))
    # (x,) = res
    # ret = (x <= edge_of_last_bucket) * (x >= -edge_of_last_bucket) * grad
    del res
    ret = grad
    return (ret, None)


def round_to_nearest_even(x: jnp.ndarray, dtype: jnp.dtype) -> jnp.ndarray:
  x = x.astype(dtype)
  # bitcast_convert to uint8 to avoid allow_excess_precision set in XLA
  x = jax.lax.bitcast_convert_type(x, jnp.uint8)
  x = jax.lax.bitcast_convert_type(x, dtype)
  return x
