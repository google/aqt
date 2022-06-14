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

"""Utilities for quantizing single tensor values.

The building block for quantized operations with multiple inputs
is quantizing individual inputs, possibly with functionally coupled parameters.
This module provides configurable functions for single-tensor calibration and
quantization.
"""

from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from aqt.common import aqt_common
from aqt.common import aqt_config
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.tpu as tpu_ops


# For compatibility with different frameworks, such as Babelfish, we allow
# custom variable creation lambdas. Make sure they're not trainable.
#
# GetVariable(name, shape, dtype, const_initial_value)
GetVariable = Callable[[str, Iterable[int], tf.dtypes.DType, Any], tf.Variable]


# Note type(default_get_variable) == GetVariable.
def default_get_variable(name: str, shape: Iterable[int],
                         dtype: tf.dtypes.DType, init: Any) -> tf.Variable:
  """Creates a non-trainable resource variable with tf.get_variable."""
  initializer = lambda: tf.constant(init, shape=shape, dtype=dtype)
  return tf.get_variable(
      name,
      dtype=dtype,
      trainable=False,
      initializer=initializer,
      use_resource=True)


class Stats:
  """Manages efficient gathering of running statistics."""

  def __init__(
      self,  #
      *,
      data_shape: Iterable[Optional[int]],
      config: aqt_config.StatsConfig,
      get_variable: GetVariable):
    self._data_shape = list(data_shape)
    config.validate(self._data_shape)
    if config.lp_order > 30:
      raise NotImplementedError('For higher norms we should add stabilization.')
    self._config = config
    self._ema_update_count = self._config.ema_update_count

    self.stats_shape = self._data_shape[:]
    for axis in self._config.share_stats_axes:
      self.stats_shape[axis] = 1

    def mk_var(name, init_val):
      return get_variable(name, self.stats_shape, tf.float32, init_val)

    self._sum_of_ones = mk_var('sum_of_ones', self._config.update_count_prior)
    self._sum_of_vals = mk_var(
        'sum_of_vals',
        self._config.mean_prior * self._config.update_count_prior)
    self._max_of_abs_vals = mk_var('max_of_abs_vals',
                                   self._config.max_dev_prior)
    self._sum_of_l1_vals = mk_var(
        'sum_of_l1_vals',
        self._config.l1_dev_prior * self._config.update_count_prior)
    self._sum_of_lp_vals = mk_var(
        'sum_of_lp_vals', self._config.lp_dev_prior**self._config.lp_order *
        self._config.update_count_prior)

  def update(self, x: tf.Tensor, weight: Optional[tf.Tensor]) -> tf.Operation:
    """Updates internal statistics with the new observation."""
    aqt_common.check_shapes_conformal(x.shape.as_list(), self._data_shape)
    if weight is not None and len(x.shape) != len(weight.shape):
      raise ValueError(
          f'expected rank(x)={len(x.shape)} == rank(weight)={weight.shape}')

    def update_var(var, s, reduce_max=False):
      assert len(s.shape) == len(self._data_shape), (s.shape, self._data_shape)
      if weight is not None:
        if reduce_max:
          # A maximum is a maximum, regardless of its nonzero weight.
          s = tf.where_v2(weight > 0, s, 0)
        else:
          s = s * weight
      reduce_fn = tf.math.reduce_max if reduce_max else tf.math.reduce_sum
      s = reduce_fn(s, axis=list(self._config.share_stats_axes), keepdims=True)
      if self._config.tpu_cross_replica_sum:
        raise NotImplementedError(
            'support for tpu_cross_replica_sum=True is not implemented')
        s = tpu_ops.cross_replica_sum(s)  # pylint:disable=unreachable
      rate = 1.0 / self._ema_update_count
      return var.assign((1.0 - rate) * var.read_value() + rate * s)

    # Layers such as ReLU emit zeros often. In such cases, we can model
    # the non-sparse distribution of weights separately, resulting in
    # unbiased estimation of non-sparse mean l1 and lp.
    # This clips away less of the distribution of inputs.
    if self._config.filter_zeros:
      ones = tf.cast(tf.math.not_equal(x, 0), dtype=tf.float32)
    else:
      ones = tf.ones_like(x)

    # Below, we need not excerpt the zeros from updates to sum of
    # vals, l1, and lp, because zeros do not affect those aggregates.

    # Avoid unnecessary tf.abs for even powers.
    px = x if self._config.lp_order % 2 == 0 else tf.abs(x)

    return tf.group([
        update_var(self._sum_of_ones, ones),
        update_var(self._sum_of_vals, x),
        update_var(self._max_of_abs_vals, tf.abs(x), reduce_max=True),
        update_var(self._sum_of_l1_vals, tf.abs(x)),
        update_var(self._sum_of_lp_vals, px**self._config.lp_order)
    ])

  def mean(self) -> tf.Tensor:
    return self._sum_of_vals.read_value() / self._sum_of_ones.read_value()

  def l1_dev(self) -> tf.Tensor:
    return self._sum_of_l1_vals.read_value() / self._sum_of_ones.read_value()

  def lp_dev(self) -> tf.Tensor:
    if self._config.lp_order == 2:
      # sqrt() is numerically more accurate
      return tf.sqrt(self._sum_of_lp_vals.read_value() /
                     self._sum_of_ones.read_value())
    else:
      # TODO(b/205769820): Make sure if the output of pow op below is
      # numerically valid
      return (self._sum_of_lp_vals.read_value() /
              self._sum_of_ones.read_value())**(1.0 / self._config.lp_order)

  def max_dev(self) -> tf.Tensor:
    return self._max_of_abs_vals

  def bound(  #
      self, calibration_config: aqt_config.CalibrationConfig) -> tf.Tensor:
    """Compute the upper bound on input tensor values, broadcastable to input."""
    return (calibration_config.l1_dev_coeff * self.l1_dev() +  #
            calibration_config.lp_dev_coeff * self.lp_dev() +  #
            calibration_config.max_dev_coeff * self.max_dev() +  #
            calibration_config.const_bound_coeff)

  def calibration_variables(self) -> Dict[str, tf.Variable]:
    """Returns variables used for self.bound()."""
    return {
        'sum_of_ones': self._sum_of_ones,  #
        'sum_of_vals': self._sum_of_vals,
        'max_of_abs_vals': self._max_of_abs_vals,
        'sum_of_l1_vals': self._sum_of_l1_vals,
        'sum_of_lp_vals': self._sum_of_lp_vals,
    }


def is_config_active(config: aqt_config.AqtTensorConfig,
                     event_count: tf.Tensor) -> tf.Tensor:
  """Return whether the configuration is active at event_count."""
  config.validate()
  should_q = tf.constant(True)
  if config.begin_at_event is not None:
    should_q &= config.begin_at_event <= event_count
  if config.end_at_event is not None:
    should_q &= event_count < config.end_at_event
  return should_q


def _should_update_scale(
    config: aqt_config.AqtTensorConfig,  #
    prev_event_count: tf.Tensor,
    new_event_count: tf.Tensor) -> tf.Tensor:
  """Returns if scale should be updated for the config and event count."""
  if isinstance(config.quant_config, aqt_config.FloatConfig):
    return tf.constant(False)

  if not config.freeze_scale_at_begin:
    return tf.constant(True)

  # The first time a config is active, even if we freeze scale, we should
  # update the scale.
  was_previously_inactive = ~is_config_active(config, prev_event_count)

  # We rely on tf.int64.min being an illegal event count value, so that
  # even if is_config_active(config, tf.int64.min), we still update scale.
  # This could happen if the very first update happens for a config which
  # has freeze_scale_at_begin and begin=None.
  assert_op = tf.debugging.assert_greater(
      new_event_count, tf.int64.min, message='event_count cannot be int64.min')
  with tf.control_dependencies([assert_op]):
    first_event = tf.math.equal(prev_event_count, tf.int64.min)

  return was_previously_inactive | first_event


class TensorQuantizer:
  """Maintains state associated with the quantization of an input tensor.

  A TensorQuantizer owns observed statistics for an input tensor, along with
  variables for the scale and event_count copy, recording the last time of the
  most recent update to this `TensorQuantizer` class.

  This class provides state-mutating methods for updating statistics for every
  observation of an input tensor and is used by AQT methods to derive
  calibration bounds.

  TensorQuantizer assumes the supplied `event_count` strictly monotonically
  increases across `TensorQuantizer.update` calls and starts out strictly
  greater than `tf.int64.min`.

  Uses the `get_variable` provided in the constructor to create variables.

  Attributes:
    data_shape: the shape of the tensor this quantizes. These dimensions may
      be `None`, but then they must correspond to shared stats axes in
      `config.stats_config`. In this case, use of quanitzed variable is
      disallowed.
    config: A training quantization schedule.
    quantized_variable: If `config.use_quantized_variable`, then the
      quantized version of the most recent `update()` tensor is saved to this
      member.  This is helpful in inference settings, where users may be
      interested in only saving quantized versions of the weights to reduce
      storage consumption or avoid quantization of floating point weights at
      inference time.
  """

  def __init__(
      self,  #
      data_shape: Iterable[int],
      config: aqt_config.AqtScheduleConfig,
      get_variable: GetVariable = default_get_variable,
      name: str = 'tensor_quantizer'):
    self.data_shape = list(data_shape)
    config.fill_gaps_with_float_config()
    config.validate(self.data_shape)
    self.config = config

    with tf.variable_scope(name):
      self._stats = Stats(
          data_shape=self.data_shape,  #
          config=self.config.stats_config,
          get_variable=get_variable)

      # We intentionally initialize scale to zero to fail loudly if someone uses
      # a parameter such as scale without properly update()-ing it.
      self._scale = get_variable('scale', self._stats.stats_shape, tf.float32,
                                 0)
      # Save the inverse scale so that we don't recompute it at inference time.
      self._inv_scale = get_variable('inv_scale', self._stats.stats_shape,
                                     tf.float32, 0)

      # Variable to save or read quantized tensors to, if the config says so.
      if self.config.use_quantized_variable:
        self.quantized_variable = get_variable('quantized_variable',
                                               self.data_shape, tf.int8, 0)

      # This variable maintains the most recent event count at which this
      # TensorQuantizer was updated. This determines which quantization config
      # from our schedule `self.config` is active, which is required because:
      # (1) at inference time, this determines the quantization config to serve.
      # (2) at training time, the first time a config with a frozen scale is
      # active, we must update the scale, a calculation requiring this variable.
      self._last_update = get_variable('last_update', [], tf.int64,
                                       tf.int64.min)

  def calibration_variables(self) -> Dict[str, tf.Variable]:
    """Returns scale and stats variables used to calibrate tensors."""
    return {
        'scale': self._scale,
        'inv_scale': self._inv_scale,
        **self._stats.calibration_variables()
    }

  def _fresh_scale(
      self, config: aqt_config.AqtTensorConfig
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Returns new scale, inverse scale for a given config and stats, if any."""
    if isinstance(config.quant_config, aqt_config.FloatConfig):
      # We shouldn't update the scale if the given config contains FloatConfig;
      # fill with a poison value if we get into this situation.
      nan = tf.constant(float('nan'), tf.float32, self._stats.stats_shape)
      return nan, nan

    x_bound = self._stats.bound(config.calibration_config)
    clip_bound = aqt_common.get_clip_bound(config.quant_config)

    new_scale = clip_bound / x_bound
    inv_scale = x_bound / clip_bound
    return new_scale, inv_scale

  def clip_range(self) -> tf.Tensor:
    """Returns the tensor clip range or zeros if no int config is active."""

    def case_fn(config: aqt_config.AqtTensorConfig) -> tf.Tensor:
      if not isinstance(config.quant_config, aqt_config.IntQuantConfig):
        return tf.zeros(self._stats.stats_shape)

      # We return the range derived from the inverse scale, rather than
      # from the stats themselves, to respect freezing settings and
      # report the clipping range that's actually used at all times.
      #
      # The counterfactual clipping range that would have been used
      # if we didn't freeze the scale can be re-derived from the current
      # stats values, which are updated regardless of freezing.
      clip_bound = aqt_common.get_clip_bound(config.quant_config)
      return self._inv_scale.read_value() * clip_bound

    return self._config_case(case_fn, self._last_update.read_value())

  def update(self,  #
             sample: tf.Tensor,
             weight: Optional[tf.Tensor],
             event_count: tf.Tensor) -> tf.Operation:
    """Op to update statistics, scale, and quantized variable.

    Args:
      sample: an observation of the tensor to quantize.
      weight: the weight of each observation for updating statistics.
      event_count: the event count this observation corresponds to; this
        determines the scale that's active for any quantized methods referencing
        this `TensorQuantizer`.

    Returns:
      A tensorflow operation corresponding to the updates to internal variables
      for capturing this observation of `sample`.
    """
    with tf.control_dependencies([self._stats.update(sample, weight)]):

      def case_fn(config):
        return self._update_state_config(config, sample, event_count)

      return self._config_case(case_fn, event_count).op

  def _config_case(
      self,  #
      case_fn: Callable[[aqt_config.AqtTensorConfig], tf.Tensor],
      event_count: tf.Tensor,
  ) -> tf.Tensor:
    """Switches over configs, applying case_fn to active one at event_count."""
    assert self.config.tensor_configs, 'There must be at least one config.'

    def make_case(config):
      pred = is_config_active(config, event_count)
      return pred, lambda: case_fn(config)

    cases = [make_case(c) for c in self.config.tensor_configs]
    return tf.case(cases, exclusive=True)

  def _update_state_config(
      self,  #
      config: aqt_config.AqtTensorConfig,
      sample: tf.Tensor,
      event_count: tf.Tensor) -> tf.Tensor:
    """Returns tensor with dependency on updates for state variables."""
    updates = []

    # Ensure we read _last_update before we update it.
    last_update = self._last_update.read_value()
    with tf.control_dependencies([last_update]):
      should_update_scale = _should_update_scale(
          config,  #
          prev_event_count=last_update,
          new_event_count=event_count)
      updates.append(self._last_update.assign(event_count))

    updated_scale, updated_inv_scale = tf.cond(
        should_update_scale,  #
        lambda: self._fresh_scale(config),
        lambda: (self._scale.read_value(), self._inv_scale.read_value()))
    updates.append(self._scale.assign(updated_scale))
    updates.append(self._inv_scale.assign(updated_inv_scale))

    if (self.config.use_quantized_variable and
        isinstance(config.quant_config, aqt_config.IntQuantConfig) and
        config.quant_config.compatible_with_int8()):
      with tf.control_dependencies(updates):
        updated_quantized_variable = tf.cast(
            self._to_quant(sample, train=True), self.quantized_variable.dtype)
        updates.append(
            self.quantized_variable.assign(updated_quantized_variable))

    with tf.control_dependencies(updates):
      return tf.constant(0)

  def _quantization_params(self, train):
    """Returns parameters for AQT quantization routine."""

    def qparams(
        config: aqt_config.AqtTensorConfig,
        check_active: bool = True
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
      """Returns quant parameters for fixed AQT config."""

      should_quantize = tf.constant(0.0)
      clip_bound = tf.constant(0.0)
      shift_before = tf.constant(0.0)
      shift_after = tf.constant(0.0)

      config_active = is_config_active(config, self._last_update.read_value())
      config_active = not check_active or config_active

      if isinstance(config.quant_config, aqt_config.FloatConfig):
        clip_bound = tf.where_v2(config_active, float('inf'), 0.0)
        return should_quantize, clip_bound, shift_before, shift_after

      config_active = tf.cast(config_active, tf.float32)
      should_quantize += config_active

      # TODO(vladf): some serving environments, such as adbrain,
      # automatically rewrite constants from f32 to bf16, which makes the
      # epsilon used in the function below invalid, so that
      # safe_clip_bound == clip_bound (incorrectly). We should solve for
      # that through some configuration.
      clip_bound += config_active * aqt_common.safe_clip_bound(
          config.quant_config)

      if config.quant_config.preserve_zero:
        shift_before += config_active * 0.5
      else:
        shift_after += config_active * 0.5

      return should_quantize, clip_bound, shift_before, shift_after

    if not train and self.config.inference_config_index is not None:
      should_quantize, clip_bound, shift_before, shift_after = qparams(
          self.config.tensor_configs[self.config.inference_config_index],
          check_active=False)
    elif self.config.tensor_configs:
      should_quantize, clip_bound, shift_before, shift_after = zip(
          *map(qparams, self.config.tensor_configs))
      should_quantize = tf.add_n(should_quantize)
      clip_bound = tf.add_n(clip_bound)
      shift_before = tf.add_n(shift_before)
      shift_after = tf.add_n(shift_after)
    else:
      should_quantize = clip_bound = shift_before = shift_after = tf.constant(
          0.0)

    should_quantize = tf.cast(should_quantize, tf.bool)
    return should_quantize, clip_bound, shift_before, shift_after

  def _to_quant(self, x: tf.Tensor, train: bool) -> tf.Tensor:
    """Quantizes x with active quant config, if any, else act as identity."""
    with tf.variable_scope('to_quant'):
      should_quantize, clip_bound, shift_before, shift_after = (
          self._quantization_params(train))

      scale = tf.where_v2(  #
          should_quantize, self._scale.read_value(),
          tf.ones_like(self._scale.read_value()))
      maybe_floor = (
          lambda y: tf.where_v2(should_quantize, tf.math.floor(y), y))

      # Note that backprop does not depend directly on the value of _last_update
      # or any_config_active; only scales and constants need to be maintained
      # and there's no branching on ops including the input tensor x. This
      # results in significant memory reduction, see cl/415355150.
      x = scale * x
      x = tf.clip_by_value(x, -clip_bound, clip_bound)
      x += shift_before
      x = tf.grad_pass_through(maybe_floor)(x)
      x += shift_after

      return x

  def _clip_mask(self, x: tf.Tensor, train: bool) -> tf.Tensor:
    """Returns which entries of x are clipped in _to_quant(x)."""
    # TODO(vladf): we should consider re-using the same clip mask
    # computation in _to_quant to derive this clip mask instead of a separate
    # method.
    with tf.variable_scope('clip_mask'):
      should_quantize, clip_bound, _, _ = (self._quantization_params(train))
      scale = tf.where_v2(  #
          should_quantize, self._scale.read_value(),
          tf.ones_like(self._scale.read_value()))
      return tf.abs(scale * x) > clip_bound

  def _from_quant_scale(self, train: bool) -> tf.Tensor:
    """Scale to dequantize the active quant config, if any, else ones."""
    with tf.variable_scope('from_quant'):
      if not train and self.config.inference_config_index is not None:
        inference_config = self.config.tensor_configs[
            self.config.inference_config_index]
        should_dequantize = tf.constant(
            isinstance(inference_config.quant_config,
                       aqt_config.IntQuantConfig))
      else:
        should_dequantize = tf.constant(False)
        for config in self.config.tensor_configs:
          if isinstance(config.quant_config, aqt_config.FloatConfig):
            continue

          config_active = is_config_active(config, self._last_update)
          should_dequantize |= config_active

      inv_scale = tf.where_v2(  #
          should_dequantize, self._inv_scale, tf.ones_like(self._inv_scale))
      return inv_scale
