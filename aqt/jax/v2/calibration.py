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
from collections.abc import Sequence
from aqt.jax.v2 import aqt_tensor
from aqt.jax.v2 import utils
from aqt.jax.v2.numerics import numerics
import jax
import jax.numpy as jnp


def ceil_to_po2(scale: jnp.ndarray) -> jnp.ndarray:
  # With floor the biggest value (we are using jnp.max) is in the range of
  # clipping and therefore have a correct gradient.
  scale = 2 ** jnp.floor(jnp.log2(jax.lax.reciprocal(scale)))
  scale = jax.lax.reciprocal(scale)
  return scale


@utils.flax_slots_kw_only_dataclass
class Calibration(abc.ABC):
  """Abstract class for scale and bias calibration."""

  # The dtype of the quantization scale and bias arrays. If not set, the arrays
  # will be in the same dtype as the input.
  dtype: None | jnp.dtype = utils.static_field(default=None)
  # Round up the calibration to power of 2 (po2).
  po2_scale: bool = utils.static_field(default=False)

  @abc.abstractmethod
  def get_scale_and_bias_and_sparsity(
      self,
      x: jnp.ndarray,
      shared_axes: None | Sequence[utils.AxisIdx],
      numerics_: numerics.AqtNumerics,
      context: None | utils.Context = None,
  ) -> tuple[list[jnp.ndarray], list[jnp.ndarray], None | jnp.ndarray]:
    """Returns the quantizaiton scale and bias for the given input tensor."""
    # NOTE: The sparsity,  has to be compatible with scale and bias.
    # The equation is defind in QTensor.quant() and QTensor.dequant() functions.
    # NOTE: The scale and bias calculation are handled by the Calibration
    # class because there is not a single order in which they should be
    # calculated. In the case of symmetric quantization, the scale depends on
    # the bias as the bias shifts the symmetric upperbound. In the case of
    # asymmetric quantization, the bias depends on the scale as the scale
    # determines how far the bias should shift the input s.t. the minimum
    # quantized value aligns with the minimum quantization bucket.
    pass

  def init_calibration(self):
    pass


@utils.flax_slots_kw_only_dataclass
class ConstantCalibration(Calibration):
  """Calibration with a constant per-tensor or per-channel value."""

  bound: jnp.ndarray | float
  bias: None | jnp.ndarray | float = None

  def get_scale_and_bias_and_sparsity(
      self,
      x: jnp.ndarray,
      shared_axes: None | Sequence[utils.AxisIdx],
      numerics_: numerics.AqtNumerics,
      context: None | utils.Context = None,
  ) -> tuple[list[jnp.ndarray], list[jnp.ndarray], None | jnp.ndarray]:
    del context
    if isinstance(self.bound, float) and self.bound <= 0.0:
      raise ValueError(f'{self.bound=} should be positive.')
    dtype = self.dtype if self.dtype is not None else x.dtype

    # TODO(yichizh): hardcode bf16 for the scales, subject to quality evaluation
    bound = self.bound
    if jnp.isscalar(bound):
      bound_shape = list(x.shape)
      for ax in shared_axes:
        bound_shape[ax] = 1
      bound = jnp.ones(bound_shape, dtype=x.dtype) * bound
    scale = bound / numerics_.get_quant_bound()
    scale = ceil_to_po2(scale) if self.po2_scale else scale

    if self.bias is None:
      bias = []
    elif jnp.isscalar(self.bias) or isinstance(self.bias, float):
      # floats are scalars, but pytype can't infer that.
      bias = [jnp.full(x.shape, self.bias, x.dtype)]
    else:
      bias = [self.bias.astype(dtype)]
    return [scale.astype(dtype)], bias, None


@utils.flax_slots_kw_only_dataclass
class AbsMaxCalibration(Calibration):
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
        'Perhaps you are using DequantMode.THIS_INPUT (fake_quant) and forgot'
        ' to set them.'
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


@utils.flax_slots_kw_only_dataclass
class AbsMeanCalibration(Calibration):
  """Simple clipping_scale * mean(abs(x) ** p) ** (1 / p) calibration.

  Attributes:
    clipping_scale: If clipping_scale < 1.0, setting
      IntSymmetric.clip_gradient=True is likely to be important.
    p: Set it to 1 for mean of absolute scaling.
  """

  clipping_scale: float
  p: float

  def get_scale_and_bias_and_sparsity(
      self,
      x: jnp.ndarray,
      shared_axes: None | Sequence[utils.AxisIdx],
      numerics_: numerics.AqtNumerics,
      context: None | utils.Context = None,
  ) -> tuple[list[jnp.ndarray], list[jnp.ndarray], None | jnp.ndarray]:
    """Calibration."""
    del context
    assert shared_axes is not None
    dtype = self.dtype if self.dtype is not None else x.dtype

    abs_sum = jnp.sum(jnp.abs(x) ** self.p, axis=shared_axes, keepdims=True)
    count = jnp.sum(x != 0.0, axis=shared_axes, keepdims=True)
    count = jnp.where(count == 0.0, jnp.ones_like(count), count)
    abs_mean = (abs_sum / count) ** (1.0 / self.p)
    abs_mean = abs_mean * self.clipping_scale
    abs_mean = jnp.where(abs_mean == 0.0, jnp.ones_like(abs_mean), abs_mean)

    scale = abs_mean / numerics_.get_quant_bound()
    scale = ceil_to_po2(scale) if self.po2_scale else scale
    return [scale.astype(dtype)], [], None


@utils.flax_slots_kw_only_dataclass
class SnrBasedAutoCalibration(Calibration):
  """Automatically finds the best clipping scales based on SNR values.

  The best clipping scales are determined by the SNR (signal-to-noise ratio)
  values of the quantized tensor. The SNR is calculated by the following
  formula:
    SNR = log(1 + signal / noise)
  where signal = sum(x ** 2) and noise = sum(err ** 2).
  An SNR value is calculated for each clipping scale per subchannel group.
  Clipping scales that produce the highest SNR value for each subchannel group
  are selected and used to calculate the best quantization scale.

  Attributes:
    auto_clip_search_config: A sequence of clipping scales to use to search for
      the best per-channel quantization scale.
  """

  auto_clip_search_config: utils.AutoScaleSearchConfig

  def get_scale_and_bias_and_sparsity(
      self,
      x: jnp.ndarray,
      shared_axes: None | Sequence[utils.AxisIdx],
      numerics_: numerics.AqtNumerics,
      context: None | utils.Context = None,
  ) -> tuple[list[jnp.ndarray], list[jnp.ndarray], None | jnp.ndarray]:
    """Produces the scale for quantization based on SNR values.

    Args:
      x: The input tensor.
      shared_axes: Axes that each subchannel group is shared across.
      numerics_: An `AqtNumerics` object containing information regarding
        quantization such as target dtype. Also used to actually quantize (round
        and clip) the tensor when calculating the SNR values.
      context: The quantization context.

    Returns:
      The scale tensor containing the scale values for each subchannel group.
      Its shape will be the same as `x.shape` but with `shared_axes` collapsed
      to 1. Biases are not supported.
    """
    dtype = self.dtype if self.dtype is not None else x.dtype

    # Determine the shape of the best_subchannel_clip_scales. There will be one
    # clip scale per subchannel group, so it shape will be the same as
    # `x.shape` but with `shared_axes` collapsed to 1.
    clip_shape = list(x.shape)
    for i in shared_axes:
      clip_shape[i] = 1

    # Default clipping scale of 1.0 (max value). One per subchannel group.
    best_subchannel_clip_scales = jnp.ones(clip_shape, dtype=jnp.float32)

    # Start with the worst possible SNR value of zeros, essentially representing
    # infinite noise. This will be updated as we search through the clip
    # scales.
    max_snr_values = jnp.zeros(clip_shape, dtype=jnp.float32)

    abs_max = jnp.max(jnp.abs(x), axis=shared_axes, keepdims=True)
    abs_max = jnp.where(abs_max == 0.0, jnp.ones_like(abs_max), abs_max)

    # Iteratively search through the clip scales search space, identifying and
    # updating the best SNR and corresponding clip scale for each subchannel
    # group. Essentially it is performing the "find the max value" for each
    # subgroup in O(auto_clip_search_config) time.
    for clip_scale in self.auto_clip_search_config:
      # Replace the new highest SNR values and corresponding clip scales
      # after evaluating for `clip`.
      best_subchannel_clip_scales, max_snr_values = (
          self._update_best_clip_scales_and_max_snr(
              best_subchannel_clip_scales,
              max_snr_values,
              clip_scale,
              x,
              abs_max,
              shared_axes,
              numerics_,
              context,
          )
      )

    # TODO(b/339746869): Generate a simple report for the clip distribution.
    bound = abs_max * best_subchannel_clip_scales
    scale = bound / numerics_.get_quant_bound()
    scale = ceil_to_po2(scale) if self.po2_scale else scale
    return [scale.astype(dtype)], [], None

  def _update_best_clip_scales_and_max_snr(
      self,
      current_clip_scales: jnp.ndarray,
      current_snr_values: jnp.ndarray,
      clip_scale: float,
      x: jnp.ndarray,
      abs_max: jnp.ndarray,
      shared_axes: Sequence[utils.AxisIdx],
      numerics_: numerics.AqtNumerics,
      context: utils.Context,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Updates the best clip scales and max SNR values given a `clip_scale`.

    Given a `clip_scale`, this function calculates the SNR value for each
    subchannel group. It then identifies the subchannel groups that have higher
    SNR values than `current_snr_values` and updates the best clip scales and
    max SNR values for those groups.

    `current_clip_scales`, `current_snr_values`, and `abs_max` are expected to
    have the same shape, which is the same as `x.shape` but with `shared_axes`
    collapsed to 1.

    `x`, `abs_max`, `shared_axes`, and `context` are only required to calculate
    the SNR values for each subchannel group.

    Args:
      current_clip_scales: The current best clip scales for each subchannel
        group.
      current_snr_values: The current best SNR values for each subchannel group.
      clip_scale: The clip scale to be evaluated.
      x: The input tensor.
      abs_max: The absolute max value for each subchannel group.
      shared_axes: Axes that each subchannel group is shared across.
      numerics_: An `AqtNumerics` object containing information regarding
        quantization such as target dtype. Also used to actually quantize (round
        and clip) the tensor when calculating the SNR values.
      context: The quantization context.

    Returns:
      The (updated best clip scales, updated best SNR values) tuple.
    """
    # Note that all subchannel groups are clipped by the same candidate clip
    # scale.
    clipped_abs_max = abs_max * clip_scale
    clipped_abs_max = clipped_abs_max.astype(x.dtype)

    snr_values = self._calculate_snr(
        x, clipped_abs_max, shared_axes, numerics_, context
    )

    # Update the best clipping scales and SNR values for subchannel groups that
    # have higher SNR values.
    updated_clip_scales = jnp.where(
        snr_values > current_snr_values,
        clip_scale,
        current_clip_scales,
    )

    updated_snr_values = jnp.maximum(snr_values, current_snr_values)
    return updated_clip_scales, updated_snr_values

  def _calculate_snr(
      self,
      x: jnp.ndarray,
      bound: jnp.ndarray,
      shared_axes: Sequence[utils.AxisIdx],
      numerics_: numerics.AqtNumerics,
      context: utils.Context,
  ) -> jnp.ndarray:
    """Calculates the quantization signal-to-noise ratio (SNR) for the given bound.

    Signal-to-noise = log(1 + signal / noise)
    where noise = sum(err ** 2) and err = x - dequantized_tensor.

    Args:
      x: The input tensor.
      bound: The bound value (scaled absolute max) for each subchannel group.
        Its shape will be the same as `x.shape` but with `shared_axes` collapsed
        to 1.
      shared_axes: Axes that each subchannel group is shared across. SNR values
        will be calculated for each dimension in `x.shape` except the shared
        axes.
      numerics_: An `AqtNumerics` object containing information regarding
        quantization such as target dtype. Also used to actually quantize (round
        and clip) the tensor when calculating the SNR values.
      context: The quantization context.

    Returns:
      The SNR tensor containing the SNR values for each subchannel group. Its
      shape will be the same as `x.shape` but with `shared_axes` collapsed to 1.
    """
    scale = bound / numerics_.get_quant_bound()
    scale = scale.astype(self.dtype if self.dtype is not None else x.dtype)

    qt = aqt_tensor.QTensor(
        qvalue=None,
        scale=[scale],
        scale_t=None,
        bias=[],
        dequant_dtype=x.dtype,
    )
    qt = qt.quant(x)

    # This actually quantizes the tensor (clips, rounds, etc).
    quantized_tensor, _ = numerics_.vjp_fwd(qt.qvalue, context)
    qt.qvalue = quantized_tensor

    dequantized_tensor = qt.dequant()
    err = x - dequantized_tensor

    noise = jnp.sum(err**2, axis=shared_axes, keepdims=True)
    signal = jnp.sum(x**2, axis=shared_axes, keepdims=True)
    snr = jnp.log(1 + signal / noise)

    return snr
