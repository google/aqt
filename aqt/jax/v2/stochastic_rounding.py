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
"""Efficient stochastic rounding implementation."""

from typing import Callable, Sequence
from aqt.jax.v2 import utils
import jax
import jax.numpy as jnp


NoiseFn = Callable[[tuple[int, ...], jax.Array], jnp.ndarray]


def _degenerate_noise_shape(
    shape: tuple[int, ...],
    noise_sharing_axes: Sequence[int] = (),
) -> tuple[int, ...]:
  """Degenerate the given shape to 1 for the broadcasting axes."""
  return tuple(
      1 if axis in noise_sharing_axes else i for axis, i in enumerate(shape)
  )


@utils.flax_slots_kw_only_dataclass
class JaxUniform:
  """Jax uniform noise."""

  def __call__(
      self,
      shape: tuple[int, ...],
      key: jax.Array,
      noise_sharing_axes: Sequence[int] = (),
  ) -> jnp.ndarray:
    noise_shape = _degenerate_noise_shape(shape, noise_sharing_axes)
    return jax.random.uniform(key, noise_shape) - 0.5


@utils.flax_slots_kw_only_dataclass
class RandomCenteredUniform:
  """Customized efficient implementation for random centered uniform noise."""

  def __call__(
      self,
      shape: tuple[int, ...],
      key: jax.Array,
      noise_sharing_axes: Sequence[int] = (),
  ) -> jnp.ndarray:
    """Generates uniform number in [-0.5, 0.5]."""
    dtype = jnp.dtype('uint16')
    nbits = jnp.iinfo(dtype).bits
    noise_shape = _degenerate_noise_shape(shape, noise_sharing_axes)

    # Generate random bits.
    bits = jax.random.bits(key, noise_shape, dtype)

    # Align bits with the mantissa of f32.
    nmant = jnp.finfo(jnp.float32).nmant
    r_bitpattern = jnp.uint32(bits) << (nmant - nbits)
    r_bitpattern = r_bitpattern | jnp.float32(1).view(jnp.uint32)
    assert r_bitpattern.dtype == jnp.uint32

    # Gen random floats and shift
    rand_floats = jax.lax.bitcast_convert_type(r_bitpattern, jnp.float32)
    shift = 2 ** (-1 - nbits)
    centered = rand_floats - (1.5 - shift)

    return centered
