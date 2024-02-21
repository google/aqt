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
"""Quantization calibration methods."""

import abc
from typing import Union
from aqt.jax.v2 import utils
import flax.linen as nn
import jax
import jax.numpy as jnp


class Calibration(abc.ABC):

  @abc.abstractmethod
  def get_bound(self, x, shared_axes) -> jnp.ndarray:
    pass


@utils.flax_slots_dataclass
class ConstantCalibration(Calibration):
  bound: Union[jnp.ndarray, float]

  def get_bound(self, x, shared_axes) -> jnp.ndarray:
    """Calibration."""
    del shared_axes
    assert self.bound > 0, 'Bound should be positive.'
    return jnp.asarray(self.bound).reshape((1,) * len(x.shape))


@utils.flax_slots_dataclass
class AbsMaxCalibration(Calibration):
  """Simple max(abs(x)) calibration."""

  def get_bound(self, x, shared_axes) -> jnp.ndarray:
    """Calibration."""
    msg = (
        'Perhaps you are using DequantMode.THIS_INPUT (fake_quant) and forgot'
        ' to set them.'
    )
    assert shared_axes is not None, msg

    # NOTE: If you want to clip, consider using clip and clip_gradient in
    # int_numerics.IntNumerics.
    abs_max = jnp.max(jnp.abs(x), axis=shared_axes, keepdims=True)
    abs_max = jnp.where(abs_max == 0.0, jnp.ones_like(abs_max), abs_max)
    return abs_max


class StaticRangeCalibration(Calibration, nn.Module):
  """State for static range AQT/PTQ/QAT."""

  moving_average_weight: float
  quant_collection: str

  @nn.compact
  def get_bound(self, x, shared_axes) -> jnp.ndarray:
    if shared_axes == 'per_tensor':
      shared_axes = list(range(x.ndim))

    running_max = jnp.max(jnp.abs(x), axis=shared_axes, keepdims=True)
    running_max = jnp.where(
        running_max == 0.0, jnp.ones_like(running_max), running_max
    )

    self.max = self.variable(
        self.quant_collection,
        'max',
        jnp.zeros,
        running_max.shape,
        running_max.dtype,
    )

    self.max.value = jax.lax.cond(
        jnp.any(self.max.value != 0 & self.max.value != 1),
        lambda: self.max.value * self.moving_average_weight
        + running_max * (1.0 - self.moving_average_weight),
        lambda: running_max,
    )

    return self.max.value
