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

import enum

from aqt.jax.v2 import calibration as aqt_calibration
import flax.linen as nn
import jax
import jax.numpy as jnp


# TODO(dhchoi): needs more proper place for this.
class QuantMode(enum.Enum):
  TRAIN = 1
  CALIBRATE = 2
  CONVERT = 3
  SERVE = 4


class StaticRangeCalibration(aqt_calibration.Calibration, nn.Module):
  """State for static range AQT/PTQ/QAT."""

  moving_average_weight: float

  # These two should be common to all static calibrations.
  quant_collection: str = "aqt"
  quant_mode: QuantMode = QuantMode.TRAIN

  @nn.compact
  def get_bound(self, x, shared_axes) -> jnp.ndarray:
    running_max = jnp.max(jnp.abs(x), axis=shared_axes, keepdims=True)
    running_max = jnp.where(
        running_max == 0.0, jnp.ones_like(running_max), running_max
    )

    self.max = self.variable(
        self.quant_collection,
        "max",
        jnp.zeros,
        running_max.shape,
        running_max.dtype,
    )

    if self.quant_mode == QuantMode.CALIBRATE:
      self.max.value = jax.lax.cond(
          jnp.any(self.max.value != 0 & self.max.value != 1),
          lambda: self.max.value * self.moving_average_weight
          + running_max * (1.0 - self.moving_average_weight),
          lambda: running_max,
      )

    return self.max.value
