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
from typing import Union
from aqt.jax.v2 import aqt_tensor
from aqt.jax.v2 import utils
from aqt.jax.v2.numerics import numerics
import jax.numpy as jnp


@utils.flax_slots_kw_only_dataclass
class Calibration(abc.ABC):
  """Abstract class for calibration."""

  @abc.abstractmethod
  def get_bound(
      self,
      x: jnp.ndarray,
      shared_axes: Sequence[utils.AxisIdx] | None,
      context: utils.Context | None = None,
  ) -> jnp.ndarray:
    pass


@utils.flax_slots_kw_only_dataclass
class ConstantCalibration(Calibration):
  """Calibration with a constant value."""

  bound: Union[jnp.ndarray, float]

  def get_bound(
      self,
      x: jnp.ndarray,
      shared_axes: Sequence[utils.AxisIdx] | None,
      context: utils.Context | None = None,
  ) -> jnp.ndarray:
    """Calibration."""
    del shared_axes, context
    assert self.bound > 0, 'Bound should be positive.'
    # TODO(yichizh): hardcode bf16 for the scales, subject to quality evaluation
    return jnp.asarray(self.bound).reshape((1,) * len(x.shape)).astype(x.dtype)


@utils.flax_slots_kw_only_dataclass
class AbsMaxCalibration(Calibration):
  """Simple max(abs(x)) calibration.

  Attributes:
    scale: Set it to something like 0.3, 0.1, 0.03. If scale < 1.0, setting
      IntNumerics.clip_gradient=True is likely to be important.
  """

  scale: float | None = None

  def get_bound(
      self,
      x: jnp.ndarray,
      shared_axes: Sequence[utils.AxisIdx] | None,
      context: utils.Context | None = None,
  ) -> jnp.ndarray:
    """Calibration.

    Args:
      x: The input tensor.
      shared_axes: Axes that share a calibration bound. For AbsMaxCalibration,
        it should not be None.
      context: The quantization context.

    Returns:
      The bound tensor containing the bound values for each group (can
      potentially be a subchannel). Its shape will be the same as `x.shape` but
      with `shared_axes` collapsed to 1.
    """
    del context

    msg = (
        'Perhaps you are using DequantMode.THIS_INPUT (fake_quant) and forgot'
        ' to set them.'
    )
    assert shared_axes is not None, msg

    # NOTE: If you want to clip, consider using clip and clip_gradient in
    # int_numerics.IntNumerics.
    abs_max = jnp.max(jnp.abs(x), axis=shared_axes, keepdims=True)
    # TODO(yichizh): the zero filtering is not needed anymore because inf is
    # filtered when calculating the reciprocal of scaline factor
    abs_max = jnp.where(abs_max == 0.0, jnp.ones_like(abs_max), abs_max)
    if self.scale is not None:
      abs_max = abs_max * self.scale
    return abs_max.astype(x.dtype)


@utils.flax_slots_kw_only_dataclass
class SnrBasedAutoCalibration(Calibration):
  """Automatically finds the best scale based on SNR values.

  The best scale is determined by the SNR (signal-to-noise ratio) values of the
  quantized tensor. The SNR is calculated by the following formula:
    SNR = log(1 + signal / noise)
  where signal = sum(x ** 2) and noise = sum(err ** 2).
  An SNR value is calculated for each scale per subchannel group. Scales that
  produce the highest SNR value for each subchannel group are selected as the
  best scale.

  Attributes:
    numerics: An `AqtNumerics` object containing information regarding
      quantization such as target dtype. Also used to actually quantize (round
      and clip) the tensor when calculating the SNR values.
    scale_search_space: A sequence of scale values, a.k.a. clipping factors, to
      search for the best scale.
  """

  numerics: numerics.AqtNumerics
  auto_scale_search_config: utils.AutoScaleSearchConfig

  def get_bound(
      self,
      x: jnp.ndarray,
      shared_axes: Sequence[utils.AxisIdx] | None,
      context: utils.Context | None = None,
  ) -> jnp.ndarray:
    """Produces the max bound for quantization based on SNR values.

    Args:
      x: The input tensor.
      shared_axes: Axes that each subchannel group is shared across.
      context: The quantization context.

    Returns:
      The bound tensor containing the bound values for each subchannel group.
      Its shape will be the same as `x.shape` but with `shared_axes` collapsed
      to 1.
    """
    # Determine the shape of the best_scale_values. There will be one scale
    # value per subchannel group, so it shape will be the same as `x.shape` but
    # with `shared_axes` collapsed to 1.
    scales_shape = list(x.shape)
    for i in shared_axes:
      scales_shape[i] = 1

    # Default value of 1.0 (max value). One scale value per subchannel group.
    best_scale_values = jnp.ones((*scales_shape,), dtype=jnp.float32)

    # Start with the worst possible SNR value of zeros, essentially representing
    # infinite noise. This will be updated as we search through the scale
    # values.
    max_snr_values = jnp.zeros((*scales_shape,), dtype=jnp.float32)

    abs_max = jnp.max(jnp.abs(x), axis=shared_axes, keepdims=True)
    abs_max = jnp.where(abs_max == 0.0, jnp.ones_like(abs_max), abs_max)

    # Iteratively search through the scale value search space, identifying and
    # updating the best SNR and corresponding scale value for each subchannel
    # group. Essentially it is performing the "find the max value" for each
    # subgroup in O(num_scales) time.
    for scale in self.auto_scale_search_config:
      # Replace the new highest SNR values and corresponding scale values
      # after evaluating for `scale`.
      best_scale_values, max_snr_values = self._update_best_scale_and_max_snr(
          best_scale_values,
          max_snr_values,
          scale,
          x,
          abs_max,
          shared_axes,
          context,
      )

    # TODO(b/339746869): Generate a simple report for the scale distribution.
    best_abs_max = abs_max * best_scale_values
    return best_abs_max.astype(x.dtype)

  def _update_best_scale_and_max_snr(
      self,
      current_scale_values: jnp.ndarray,
      current_snr_values: jnp.ndarray,
      scale: float,
      x: jnp.ndarray,
      abs_max: jnp.ndarray,
      shared_axes: Sequence[utils.AxisIdx],
      context: utils.Context,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Updates the best scale and max SNR values given a `scale` value.

    Given a `scale` value, this function calculates the SNR value for each
    subchannel group. It then identifies the subchannel groups that have higher
    SNR values than `current_snr_values` and updates the best scale and max SNR
    values for those groups.

    `current_scale_values`, `current_snr_values`, and `abs_max` are expected to
    have the same shape, which is the same as `x.shape` but with `shared_axes`
    collapsed to 1.

    `x`, `abs_max`, `shared_axes`, and `context` are only required to calculate
    the SNR values for each subchannel group.

    Args:
      current_scale_values: The current best scale values for each subchannel
        group.
      current_snr_values: The current best SNR values for each subchannel group.
      scale: The scale value to be evaluated.
      x: The input tensor.
      abs_max: The absolute max value for each subchannel group.
      shared_axes: Axes that each subchannel group is shared across.
      context: The quantization context.

    Returns:
      The (updated best scale values, updated best SNR values) tuple.
    """
    # Note that all subchannel groups are scaled by the same candidate scale
    # value.
    scaled_abs_max = abs_max * scale
    scaled_abs_max = scaled_abs_max.astype(x.dtype)

    snr_values = self._calculate_snr(x, scaled_abs_max, shared_axes, context)

    # Identify subchannel groups that have higher SNR values.
    new_snr_highs = jnp.where(
        snr_values > current_snr_values,
        jnp.ones_like(current_snr_values),
        jnp.zeros_like(current_snr_values),
    )

    updated_scale_values = jnp.where(
        new_snr_highs > 0.0,
        scale,
        current_scale_values,
    )

    updated_snr_values = jnp.where(
        new_snr_highs > 0.0,
        snr_values,
        current_snr_values,
    )
    return updated_scale_values, updated_snr_values

  def _calculate_snr(
      self,
      x: jnp.ndarray,
      bound: jnp.ndarray,
      shared_axes: Sequence[utils.AxisIdx],
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
      context: The quantization context.

    Returns:
      The SNR tensor containing the SNR values for each subchannel group. Its
      shape will be the same as `x.shape` but with `shared_axes` collapsed to 1.
    """
    abs_max_mapped_to = self.numerics.abs_val_mapped_to()
    scale = bound / abs_max_mapped_to

    q_tensor = aqt_tensor.QTensor(
        qvalue=None, scale=[scale], scale_t=None, dequant_dtype=x.dtype
    ).quant(x)

    # This actually quantizes the tensor (clips, rounds, etc).
    quantized_tensor, _ = self.numerics.vjp_fwd(q_tensor.qvalue, context)
    q_tensor = q_tensor.replace(qvalue=quantized_tensor)

    dequantized_tensor = q_tensor.dequant()
    err = x - dequantized_tensor

    noise = jnp.sum(err**2, axis=shared_axes, keepdims=True)
    signal = jnp.sum(x**2, axis=shared_axes, keepdims=True)
    snr = jnp.log(1 + signal / noise)

    return snr
