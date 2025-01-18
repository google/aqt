# Copyright 2025 Google LLC
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
"""Quantized jax.lax.ragged_dot."""

from aqt.jax.v2 import aqt_dot_general
from aqt.jax.v2 import config
from aqt.jax.v2 import utils
import jax
import jax.numpy as jnp


@utils.flax_slots_kw_only_dataclass
class RaggedDot:
  """Flax slot for jax.lax.ragged_dot.

  Attributes:
    group_sizes: The sizes of the groups with shape [g].
  """
  group_sizes: jnp.ndarray

  def __call__(
      self,
      lhs,
      rhs,
      dimension_numbers,
      precision=None,
      preferred_element_type=None,
  ):
    del dimension_numbers
    return jax.lax.ragged_dot(
        lhs,
        rhs,
        self.group_sizes,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )


def ragged_dot(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: jnp.dtype | None = None,
    cfg: aqt_dot_general.DotGeneral = config.config_v4(),
):
  """Quantized version of jax.lax.ragged_dot.

  Currently only forward pass is supported. TODO(pfilipiuk): Add support for
  the backward pass.

  Args:
    lhs: The left operand of the ragged dot with shape [m, k].
    rhs: The right operand of the ragged dot with shape [g, k, n].
    group_sizes: The sizes of the groups with shape [g].
    precision: The precision of the computation. It should not be set when
      quantization is enabled.
    preferred_element_type: The preferred element type of the result. It is
      ignored.
    cfg: The configuration of the dot general.

  Returns:
    The result of the ragged dot product.
  """
  cfg.fwd.dot_general = RaggedDot(group_sizes=group_sizes)
  # We want to use the same scale for all the experts in the rhs. We do this by
  # setting the rhs_ca to [0, 1] even though the actual contraction axis is 1.
  dimension_numbers = ([1], [0, 1]), ([], [])
  return cfg(
      lhs,
      rhs,
      dimension_numbers=dimension_numbers,
      precision=precision,
      preferred_element_type=preferred_element_type,
  )
