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

from typing import Optional
from aqt.jax.v2 import stochastic_rounding
import flax.struct
from jax import lax
import jax.numpy as jnp


class Fp8Numerics(flax.struct.PyTreeNode):
  """Numerics for float8_e4m3fn, float8_e5m2, etc."""

  exponent_bits: int
  mantissa_bits: int
  clip: bool = True
  round: bool = True
  noise_fn: Optional[stochastic_rounding.NoiseFn] = None

  def _get_edge_of_last_fp8_bucket(self):
    # TODO(lew): Currently what this function returns is actually the center of
    # the last bucket. Since the bucket size is relatively small for fp8, we now
    # keep it as it is, but it may needs to be fixed.
    if self.exponent_bits == 4 and self.mantissa_bits == 3:
      return jnp.finfo(jnp.float8_e4m3fn).max.astype(jnp.bfloat16)
    elif self.exponent_bits == 5 and self.mantissa_bits == 2:
      return jnp.finfo(jnp.float8_e5m2).max.astype(jnp.bfloat16)
    else:
      raise ValueError

  def abs_val_mapped_to(self):
    return self._get_edge_of_last_fp8_bucket()

  def fwd(self, x, context):
    edge_of_last_bucket = self._get_edge_of_last_fp8_bucket()
    # Maybe noise
    if self.noise_fn:
      assert context.key is not None, (
          'noise_fn is set, requestic stochastic rounding, but RNG was not '
          'passed in Context.key'
      )
      x = (x + self.noise_fn(x.shape, context.key)).astype(x.dtype)  # pylint: disable=not-callable

    # Maybe clip
    if self.clip:
      x = jnp.clip(x, -edge_of_last_bucket, edge_of_last_bucket)

    # Maybe round
    if self.round:
      x = lax.round(x, lax.RoundingMethod.TO_NEAREST_EVEN)

    return x

  def vjp_fwd(self, x, context):
    res = (x,)
    return self.fwd(x, context), res

  def vjp_bwd(self, res, grad):
    # This is gradient of clip. For boundary values we will have full graindent.
    # We might use something like this for calibrations other than abs(max(x))
    # (x,) = res
    # ret = (x <= edge_of_last_bucket) * (x >= -edge_of_last_bucket) * grad
    # TODO(lew): Consider clipping.
    del res
    ret = grad
    return (ret, None)
