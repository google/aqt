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
"""Static range calibration."""

from typing import Sequence

from aqt.jax.v2 import calibration
from aqt.jax.v2 import utils
import flax.linen as nn
from jax import numpy as jnp


@utils.flax_slots_dataclass
class MeanOfAbsMaxCalibration(calibration.Calibration, nn.Module):
  """State for static range AQT/PTQ/QAT."""

  quant_collection: str

  @nn.compact
  def get_bound(
      self,
      x: jnp.ndarray,
      shared_axes: Sequence[int] | None,
      context: utils.Context | None = None,
  ) -> jnp.ndarray:
    abs_max = jnp.max(jnp.abs(x), axis=shared_axes, keepdims=True)
    quant_mode = context.quant_mode if context else utils.QuantMode.TRAIN

    # Initialize the max_stat.value as 0; does not update the max_stat value
    # during initialization.
    is_initializing = not self.has_variable(self.quant_collection, "sum_of_max")

    sum_of_max = self.variable(
        self.quant_collection,
        "sum_of_max",
        jnp.zeros,
        abs_max.shape,
        abs_max.dtype,
    )
    count = self.variable(
        self.quant_collection, "count", jnp.zeros, (), jnp.int32
    )

    if quant_mode == utils.QuantMode.CALIBRATE and not is_initializing:
      sum_of_max.value = sum_of_max.value + abs_max
      count.value = count.value + 1

    # TODO(dhchoi): Introduce checker for count.value.
    # This assertion should be done on jitted graph, and this is not
    # straightforward (or does not have a good option).
    # One option is jax.experimental.checkify
    # (https://jax.readthedocs.io/en/latest/debugging/checkify_guide.html), but
    # we need to transform the whole function with jax.experimental.checkify to
    # use this feature.
    # Maybe wait for the JAX language upgrade to have a better support for this?
    return sum_of_max.value / count.value
