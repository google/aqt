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
"""State Updator class to collect the running statistics."""

import abc
from typing import Sequence

from aqt.jax.v2 import aqt_state
from aqt.jax.v2 import calibration as aqt_calibration
from aqt.jax.v2 import config
import flax.linen as nn
import jax
import jax.numpy as jnp


class DotGeneralStateUpdator(abc.ABC, nn.Module):
  """Abstract class for managing dot_general's running statistics.

  Each algorithms should inherit this class to implement its own.
  """
  cfg: config.DotGeneralRaw

  is_lhs: bool
  lhs_shape: Sequence[int]
  rhs_shape: Sequence[int]
  lhs_dtype: jnp.dtype
  rhs_dtype: jnp.dtype
  dimension_numbers: jax.lax.DotDimensionNumbers

  quant_collection: str

  @abc.abstractmethod
  @nn.compact
  def update(self, lhs: jnp.ndarray, rhs: jnp.ndarray) -> None:
    """Updates the statistics values.

    We get lhs and rhs as its inputs, since we do not know which statistics the
    future algorithm developers are willing to collect.

    Args:
      lhs: left hand side input.
      rhs: right hand side input.
    """
    ...

  @abc.abstractmethod
  def get_state(self) -> aqt_state.State:
    """Returns state values."""
    ...


class DotGeneralStaticRangeStateUpdator(DotGeneralStateUpdator):
  """Updator to collect dot_general states for static range AQT/PTQ/QAT."""

  moving_average_weight: float

  def setup(self):
    (_, rhs_ca), _ = self.dimension_numbers
    if self.is_lhs:
      # Collection of activation statistics for static range PTQ requires the
      # channelwise configuration to be per-tensor.
      # Assuming that lhs is activation.
      assert self.cfg.lhs.quantizer.calib_shared_axes == "per_tensor"
      shared_axis = list(range(len(self.lhs_shape)))
      max_shape = self.cfg.lhs.quantizer.calibration.get_bound(
          jnp.zeros(self.lhs_shape), shared_axis
      ).shape
      dtype = self.lhs_dtype
    else:
      max_shape = self.cfg.rhs.quantizer.calibration.get_bound(
          jnp.zeros(self.rhs_shape), rhs_ca
      ).shape
      dtype = self.rhs_dtype

    self.max = self.variable(
        self.quant_collection, "max", jnp.zeros, max_shape, dtype
    )

  def update(self, lhs: jnp.ndarray, rhs: jnp.ndarray) -> None:
    (_, rhs_ca), _ = self.dimension_numbers
    if self.is_lhs:
      # Collection of activation statistics for static range PTQ requires the
      # channelwise configuration to be per-tensor.
      # Assuming that lhs is activation.
      assert self.cfg.lhs.quantizer.calib_shared_axes == "per_tensor"
      shared_axis = list(range(lhs.ndim))
      self._update(self.cfg.lhs.quantizer.calibration, lhs, shared_axis)
    else:
      self._update(self.cfg.rhs.quantizer.calibration, rhs, rhs_ca)

  def _update(
      self,
      calibration: aqt_calibration.Calibration,
      x: jnp.ndarray,
      ca: Sequence[int],
  ):
    running_max = calibration.get_bound(x, ca)

    # Prevent the max value from being calibrated with initial values 0.
    self.max.value = jax.lax.cond(
        jnp.any((self.max.value != 0) & (self.max.value != 1)),
        lambda: self.max.value * self.moving_average_weight
        + running_max * (1.0 - self.moving_average_weight),
        lambda: running_max,
    )

  def get_state(self) -> aqt_state.StaticRangeState:
    return aqt_state.StaticRangeState(self.max.value)

