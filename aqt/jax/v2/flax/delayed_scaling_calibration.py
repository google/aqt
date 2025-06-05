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
"""AQT calibration module with logic from Transformer Engine."""
from typing import Sequence

from aqt.jax.v2 import calibration
from aqt.jax.v2 import utils
from aqt.jax.v2.numerics import numerics
import flax.linen as nn
import jax
from jax import numpy as jnp

CALIBRATION_STATS = "calibration_stats"


@utils.flax_slots_kw_only_dataclass
class DelayedScalingCalibration(calibration.Calibration, nn.Module):
  """Calibration module with logic from Transformer Engine, utilizing Delayed Scaling."""

  amax_history_length: int = 1024

  def setup(self) -> None:
    # If we use nn.compact, we have to have a dummy call to self.get_bound()
    # during initialization. This is particularly problematic when using this
    # module for backward pass calibration, when get_bound() isn't naturally
    # invoked during the dummy forward pass during init().

    # Using setup() keeps things simpler and avoids the need to make awkward
    # dummy calls to get_bound().
    self.amax_history = self.variable(
        CALIBRATION_STATS,
        "amax_history",
        # pylint: disable-next=protected-access
        lambda: jnp.zeros((self.amax_history_length,)),
    )

    self.bound = self.variable(
        CALIBRATION_STATS,
        "bound",
        # pylint: disable-next=protected-access
        lambda: jnp.zeros((1,)),
    )

  def get_bound(
      self,
      x: jnp.ndarray,
      shared_axes: None | Sequence[utils.AxisIdx],
      context: None | utils.Context = None,
  ) -> jnp.ndarray:
    del shared_axes
    # Right now we just support per_tensor calibration (i.e. one value).
    # To support per_axis calibration, we would need to be able to change the
    # shape of the mutable arrays. For example, right now amax_history has
    # shape (amax_history_length,). But if we want to support per_axis
    # calibration, we would need the shape of amax_history to be changed to
    # (amax_history_length, scale_shape), where scale_shape is
    # jnp.max(jnp.abs(x), axis=shared_axes, keepdims=True).shape
    #
    # See setup() for why we can't use nn.compact to solve this as is usually
    # done Jax/Flax

    # We default to SERVE mode if context is not provided, since we will
    # be mutating the arrays in place and don't want to do so accidentally
    quant_mode = context.quant_mode if context else utils.QuantMode.SERVE

    prev_bound = self.bound.value
    amax_history = self.amax_history.value

    amax_from_history = jnp.max(amax_history, axis=0)

    new_bound = self.compute_bound(amax_from_history, prev_bound)
    new_history = self.compute_history(x, amax_history)

    return new_bound.reshape((1,) * len(x.shape))

  def get_scale_and_bias_and_sparsity(
      self,
      x: jnp.ndarray,
      shared_axes: None | Sequence[utils.AxisIdx],
      numerics_: numerics.AqtNumerics,
      context: None | utils.Context = None,
  ) -> tuple[list[jnp.ndarray], list[jnp.ndarray], None | jnp.ndarray]:
    dtype = self.dtype if self.dtype is not None else x.dtype
    bound = self.get_bound(x, shared_axes, context)
    scale = bound / numerics_.get_quant_bound()
    scale = calibration.ceil_to_po2(scale) if self.po2_scale else scale
    return [scale.astype(dtype)], [], None

  def compute_bound(self, amax, prev_bound):
    new_bound = jnp.copy(amax)
    new_bound = jnp.where(amax > 0.0, new_bound, prev_bound)
    new_bound = jnp.where(jnp.isfinite(amax), new_bound, prev_bound)
    return new_bound

  def compute_history(self, x, amax_history):
    amax_update = jnp.max(jnp.abs(x)).astype(amax_history.dtype)
    new_history = (
        jnp.roll(amax_history, shift=-1, axis=0).at[0].set(amax_update)
    )
    return new_history

  def init_calibration(self):
    # We need to "touch" one of these variables to make sure this modules
    # variables are initialized properly once. Else, they get "recreated"
    # on each use.
    self.amax_history  # pylint: disable=pointless-statement


def ceil_to_po2(scale: jnp.ndarray) -> jnp.ndarray:
  # With floor the biggest value (we are using jnp.max) is in the range of
  # clipping and therefore have a correct gradient.
  scale = 2 ** jnp.floor(jnp.log2(jax.lax.reciprocal(scale)))
  scale = jax.lax.reciprocal(scale)
  return scale


@utils.flax_slots_kw_only_dataclass
class AbsMaxCalibration(calibration.Calibration):
  """Simple max(abs(x)) calibration.

  Attributes:
    clipping_scale: Set it to something like 0.3, 0.1, 0.03. If clipping_scale <
      1.0, setting IntSymmetric.clip_gradient=True is likely to be important.
  """

  clipping_scale: None | float = None

  def get_scale_and_bias_and_sparsity(
      self,
      x: jnp.ndarray,
      shared_axes: None | Sequence[utils.AxisIdx],
      numerics_: numerics.AqtNumerics,
      context: None | utils.Context = None,
  ) -> tuple[list[jnp.ndarray], list[jnp.ndarray], None | jnp.ndarray]:
    """Calibration.

    Args:
      x: The input tensor.
      shared_axes: Axes that share a calibration bound. For AbsMaxCalibration,
        it should not be None.
      numerics_: An `AqtNumerics` object containing information regarding
        quantization. Used to create the scale and bias arrays.
      context: The quantization context.

    Returns:
      The scale tensor containing the scale values for each group (can
      potentially be a subchannel). Its shape will be the same as `x.shape` but
      with `shared_axes` collapsed to 1. Bias is not supported.
    """
    del context
    msg = (
        "Perhaps you are using DequantMode.THIS_INPUT (fake_quant) and forgot"
        " to set them."
    )
    assert shared_axes is not None, msg
    dtype = self.dtype if self.dtype is not None else x.dtype

    # NOTE: If you use a clipping_scale, consider using clip and clip_gradient
    # in int_numerics.IntSymmetric.
    abs_max = jnp.max(jnp.abs(x), axis=shared_axes, keepdims=True)
    # TODO(yichizh): the zero filtering is not needed anymore because inf is
    # filtered when calculating the reciprocal of scaling factor
    abs_max = jnp.where(abs_max == 0.0, jnp.ones_like(abs_max), abs_max)
    bound = abs_max * self.clipping_scale if self.clipping_scale else abs_max

    scale = bound / numerics_.get_quant_bound()
    scale = ceil_to_po2(scale) if self.po2_scale else scale
    return [scale.astype(dtype)], [], None
