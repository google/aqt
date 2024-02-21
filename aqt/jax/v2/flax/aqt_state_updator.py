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
"""State class to gather running statistics."""

import abc
import enum
from typing import Literal, Sequence

from aqt.jax.v2 import aqt_state
import flax.linen as nn
import jax
import jax.numpy as jnp


# TODO(dhchoi): Find a better place for this.
class QuantMode(enum.Enum):
  TRAIN = 1
  CALIBRATE = 2
  CONVERT = 3
  SERVE = 4


# 1. Define StateUpdator, which will be in charge of managing & updating
# calibrated status (min / max, for example).
#  Differently from AQTv1, StateUpdator and State (which is the mere container
# for the calibrate values) are separated for below reasons:
# (1) Remove cyclic dependency - State values should be used in quantizer, while
# state update logic highly likely depends on the quantizer logic.
# (2) Separate flax module with core dot_general (which is more jax-like) logic.


class StateUpdator(abc.ABC, nn.Module):
  """Abstract class for managing running statistics.

  Each algorithms should inherit this class to implement its own.
  """
  quant_collection: str
  quant_mode: QuantMode

  # 3. Update logic gets lhs and rhs as its input, and can do whatever it needs.
  @abc.abstractmethod
  @nn.compact
  def update(self, lhs: jnp.ndarray, rhs: jnp.ndarray) -> None:
    """Updates the statistics values.

    We get the whole configuration, lhs and rhs as its inputs, since we do not
    know which statistics the future algorithm developers are planning to
    collect.

    Args:
      lhs: left hand side input.
      rhs: right hand side input.
    """
    ...

  @abc.abstractmethod
  def get_state(self) -> aqt_state.State:
    """Returns state values."""
    ...


class DotGeneralStaticRangeStateUpdator(StateUpdator):
  """State for static range AQT/PTQ/QAT."""

  is_lhs: bool
  lhs_shape: Sequence[int]
  rhs_shape: Sequence[int]
  lhs_dtype: jnp.dtype
  rhs_dtype: jnp.dtype

  calib_shared_axes: list[int] | Literal["per_tensor"] | None
  dimension_numbers: jax.lax.DotDimensionNumbers

  moving_average_weight: float

  def setup(self):
    (_, rhs_ca), _ = self.dimension_numbers
    if self.is_lhs:
      # Collection of activation statistics for static range PTQ requires the
      # channelwise configuration to be per-tensor.
      # Assuming that lhs is activation.
      assert self.calib_shared_axes == "per_tensor"
      shared_axis = list(range(len(self.lhs_shape)))
      max_shape = self._get_max(jnp.zeros(self.lhs_shape), shared_axis).shape
      dtype = self.lhs_dtype
    else:
      max_shape = self._get_max(jnp.zeros(self.rhs_shape), rhs_ca).shape
      dtype = self.rhs_dtype

    self.max = self.variable(
        self.quant_collection, "max", jnp.zeros, max_shape, dtype
    )

  def update(self, lhs: jnp.ndarray, rhs: jnp.ndarray) -> None:
    if self.quant_mode != QuantMode.CALIBRATE:
      return

    (_, rhs_ca), _ = self.dimension_numbers
    if self.is_lhs:
      # Collection of activation statistics for static range PTQ requires the
      # channelwise configuration to be per-tensor.
      # Assuming that lhs is activation.
      assert self.calib_shared_axes == "per_tensor"
      shared_axis = list(range(lhs.ndim))
      self._update(lhs, shared_axis)
    else:
      self._update(rhs, rhs_ca)

  def _get_max(self, x: jnp.ndarray, ca: Sequence[int]):
    abs_max = jnp.max(jnp.abs(x), axis=ca, keepdims=True)
    abs_max = jnp.where(abs_max == 0.0, jnp.ones_like(abs_max), abs_max)
    return abs_max

  def _update(self, x: jnp.ndarray, ca: Sequence[int]):
    running_max = self._get_max(x, ca)

    self.max.value = jax.lax.cond(
        jnp.any((self.max.value != 0) & (self.max.value != 1)),
        lambda: self.max.value * self.moving_average_weight
        + running_max * (1.0 - self.moving_average_weight),
        lambda: running_max,
    )

  def get_state(self) -> aqt_state.StaticRangeState | None:
    if self.quant_mode == QuantMode.TRAIN:
      return None

    if self.quant_mode == QuantMode.SERVE and not self.is_lhs:
      return None

    return aqt_state.StaticRangeState(self.max.value)

