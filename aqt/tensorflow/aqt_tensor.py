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
quantization, including static quantization allowing freeze quantization scales
and dynamic quantization with history-independent quantization scales.
quantization.
"""
import dataclasses
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from aqt.common import aqt_common
from aqt.common import aqt_config
from aqt.common import emulated_floating_points
from aqt.common import emulation_utils
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.tpu as tpu_ops


def _emulated_fp(
    t: tf.Tensor,
    mantissa_bits: tf.Tensor,
    min_exp: tf.Tensor,
    max_exp: tf.Tensor,
) -> tf.Tensor:
  """Emulates enumerics in bfloat16 or float32 given the FPMetadata.

  This is based on the implementation in:
  google3/platforms/deepsea/ffds/reduced_precision/emulated_floating_points.py
  However a separate implementation is needed to incorporate the metadata as
  tensors in the TF graph.

  Args:
    t: a tensor whose dtype is either tf.bfloat16 or tf.float32.
    mantissa_bits: The mantissa bits in the emulated format.
    min_exp: the allowed minimum exponents in the emulated format.
    max_exp: the allowed maximum exponents in the emulated format.

  Returns:
    a same dtype tensor that emulates floating points.
  """
  assert t.dtype == tf.float32
  output_type = t.dtype

  with tf.name_scope('emulated_fp'):
    v = emulated_floating_points.handle_mantissa(
        t,
        mantissa_bits=mantissa_bits,
        min_exp=min_exp,
        rounding_mode=emulation_utils.ROUND_TO_NEAREST_EVEN)
    v = emulated_floating_points.static_handle_exponent(
        v,
        min_exp=min_exp - mantissa_bits,
        max_exp=max_exp,
        mantissa_bits=mantissa_bits)
    return tf.cast(v, output_type)

# For compatibility with different frameworks, such as Babelfish, we allow
# custom variable creation lambdas. Make sure they're not trainable.
#
# GetVariable(name, shape, dtype, const_initial_value)
GetVariable = Callable[[str, Iterable[int], tf.dtypes.DType, Any], tf.Variable]


# Note type(default_get_variable) == GetVariable.
def default_get_variable(name: str, shape: Iterable[int],
                         dtype: tf.dtypes.DType, init: Any) -> tf.Variable:
  """Creates a non-trainable resource variable with tf.get_variable."""
  initializer = lambda shape, dtype: tf.constant(init, shape=shape, dtype=dtype)
  return tf.get_variable(
      name=name,
      shape=shape,
      dtype=dtype,
      trainable=False,
      initializer=initializer,
      use_resource=True)


_DivideFn = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]


def _lp_dev(
    sum_of_lp_vals: tf.Tensor,  #
    sum_of_ones: tf.Tensor,
    lp_order: int,
    divide_fn: _DivideFn = tf.math.divide_no_nan,
) -> tf.Tensor:
  """Computes lp deviation."""
  if lp_order == 2:
    # sqrt() is numerically more accurate
    return tf.sqrt(divide_fn(sum_of_lp_vals, sum_of_ones))
  else:
    # TODO(b/205769820): Make sure if the output of pow op below is
    # numerically valid
    return divide_fn(sum_of_lp_vals, sum_of_ones) ** (1.0 / lp_order)


def _reduce_fn(
    stats_config: aqt_config.StatsConfig,
    s: tf.Tensor,
    weight: Optional[tf.Tensor],
    reduce_max=False,
) -> tf.Tensor:
  """Reduce function to calculate statistics."""
  if weight is not None:
    if reduce_max:
      # A maximum is a maximum, regardless of its nonzero weight.
      s = tf.where_v2(weight > 0, s, 0)
    else:
      s = s * weight
  reduce_fn = tf.math.reduce_max if reduce_max else tf.math.reduce_sum
  s = reduce_fn(s, axis=list(stats_config.share_stats_axes), keepdims=True)
  if stats_config.tpu_cross_replica_sum:
    raise NotImplementedError(
        'support for tpu_cross_replica_sum=True is not implemented'
    )
    s = tpu_ops.cross_replica_sum(s)  # pylint:disable=unreachable
  return s


def _sum_of_ones(
    stats_config: aqt_config.StatsConfig,
    x: tf.Tensor,
    weight: Optional[tf.Tensor],
) -> tf.Tensor:
  """Computes the number of entries along non-shared stats axes (maybe filter zeros)."""
  # Layers such as ReLU emit zeros often. In such cases, we can model
  # the non-sparse distribution of weights separately, resulting in
  # unbiased estimation of non-sparse mean l1 and lp.
  # This clips away less of the distribution of inputs.
  if stats_config.filter_zeros:
    ones = tf.cast(tf.math.not_equal(x, 0), dtype=tf.float32)
  else:
    ones = tf.ones_like(x)
  return _reduce_fn(stats_config, ones, weight)


def _max_of_abs_vals(
    stats_config: aqt_config.StatsConfig,
    x: tf.Tensor,
    weight: Optional[tf.Tensor],
) -> tf.Tensor:
  return _reduce_fn(stats_config, tf.abs(x), weight, reduce_max=True)


_sum_of_vals = _reduce_fn


def _sum_of_l1_vals(
    stats_config: aqt_config.StatsConfig,
    x: tf.Tensor,
    weight: Optional[tf.Tensor],
) -> tf.Operation:
  return _reduce_fn(stats_config, tf.abs(x), weight)


def _sum_of_lp_vals(
    stats_config: aqt_config.StatsConfig,
    x: tf.Tensor,
    weight: Optional[tf.Tensor],
) -> tf.Operation:
  # Below, we need not excerpt the zeros from updates to sum of
  # vals, l1, and lp, because zeros do not affect those aggregates.

  # Avoid unnecessary tf.abs for even powers.
  px = x if stats_config.lp_order % 2 == 0 else tf.abs(x)

  return _reduce_fn(stats_config, px**stats_config.lp_order, weight)


def get_stats_shape(
    share_stats_axes: Iterable[int], data_shape: Iterable[Optional[int]]
) -> List[Optional[int]]:
  """Returns the shape of the statistics.

  Replaces the dimensions in the data shape with ones where we share statistics.

  Args:
    share_stats_axes: axes where statistics are shared.
    data_shape: shape of a tensor.

  Returns:
    the shape of statistics.
  """
  stats_shape = list(data_shape)
  for axis in share_stats_axes:
    stats_shape[axis] = 1
  return stats_shape


def _init_dynamic_stats(
    stats_config: aqt_config.StatsConfig,
    x: tf.Tensor,
    init_value: float = 0.0,
) -> tf.Tensor:
  """Initializes a dynamic statistical tensor."""
  # assume x is of dynamic shape and we want to have a constant tensor with the
  # shape of x except for shared statistics axes where dimensions are ones.
  rank = len(x.shape.as_list())
  indices = []
  for i in range(rank):
    if i in stats_config.share_stats_axes:
      indices.append(slice(0, 1))
    else:
      indices.append(slice(None))
  ones = tf.ones_like(x[indices], dtype=x.dtype)
  return ones * init_value


def _bound(
    calibration_config: aqt_config.CalibrationConfig,
    lp_order: int,
    init_bound: tf.Tensor,
    sum_of_ones: Optional[tf.Tensor],
    max_of_abs_vals: Optional[tf.Tensor],
    sum_of_l1_vals: Optional[tf.Tensor],
    sum_of_lp_vals: Optional[tf.Tensor],
    divide_fn: _DivideFn,
) -> tf.Tensor:
  """Computes the upper bound."""
  bound = init_bound + calibration_config.const_bound_coeff
  if calibration_config.l1_dev_coeff:
    l1_dev = divide_fn(sum_of_l1_vals, sum_of_ones)
    bound += calibration_config.l1_dev_coeff * l1_dev
  if calibration_config.lp_dev_coeff:
    lp_dev = _lp_dev(sum_of_lp_vals, sum_of_ones, lp_order, divide_fn)
    bound += calibration_config.lp_dev_coeff * lp_dev
  if calibration_config.max_dev_coeff:
    max_dev = max_of_abs_vals
    bound += calibration_config.max_dev_coeff * max_dev
  return bound


def _dynamic_bound(
    config: aqt_config.StatsConfig,
    calibration_config: aqt_config.CalibrationConfig,
    x: tf.Tensor,
    weight: Optional[tf.Tensor],
) -> tf.Tensor:
  """Compute the upper bound on input tensor values dynamically."""
  config.validate(x.shape.as_list(), dynamic=True)
  init_bound = _init_dynamic_stats(config, x, init_value=0.0)
  divide_fn = tf.math.divide_no_nan if config.safe_divide else tf.divide
  sum_of_ones = max_of_abs_vals = sum_of_l1_vals = sum_of_lp_vals = None
  if any([
      calibration_config.l1_dev_coeff,
      calibration_config.lp_dev_coeff,
      calibration_config.max_dev_coeff,
  ]):
    sum_of_ones = _sum_of_ones(config, x, weight)
  if calibration_config.max_dev_coeff:
    max_of_abs_vals = _max_of_abs_vals(config, x, weight)
  if calibration_config.l1_dev_coeff:
    sum_of_l1_vals = _sum_of_l1_vals(config, x, weight)
  if calibration_config.lp_dev_coeff:
    sum_of_lp_vals = _sum_of_lp_vals(config, x, weight)
  return _bound(
      calibration_config,
      config.lp_order,
      init_bound,
      sum_of_ones,
      max_of_abs_vals,
      sum_of_l1_vals,
      sum_of_lp_vals,
      divide_fn,
  )


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

    self.stats_shape = get_stats_shape(
        self._config.share_stats_axes,
        self._data_shape,
    )

    self.divide = (tf.math.divide_no_nan if self._config.safe_divide
                   else tf.math.divide)

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
          f'expected rank(x)={len(x.shape)} == rank(weight)={len(weight.shape)}'
      )

    def update_var(var, update_fn):
      s = update_fn(self._config, x, weight)
      rate = 1.0 / self._ema_update_count
      return var.assign((1.0 - rate) * var.read_value() + rate * s)

    return tf.group([
        update_var(self._sum_of_ones, _sum_of_ones),
        update_var(self._sum_of_vals, _sum_of_vals),
        update_var(self._max_of_abs_vals, _max_of_abs_vals),
        update_var(self._sum_of_l1_vals, _sum_of_l1_vals),
        update_var(self._sum_of_lp_vals, _sum_of_lp_vals),
    ])

  def mean(self) -> tf.Tensor:
    return self.divide(self._sum_of_vals.read_value(),
                       self._sum_of_ones.read_value())

  def l1_dev(self) -> tf.Tensor:
    return self.divide(self._sum_of_l1_vals.read_value(),
                       self._sum_of_ones.read_value())

  def lp_dev(self) -> tf.Tensor:
    return _lp_dev(
        self._sum_of_lp_vals.read_value(),
        self._sum_of_ones.read_value(),
        self._config.lp_order,
        self.divide,
    )

  def max_dev(self) -> tf.Tensor:
    return self._max_of_abs_vals

  def bound(  #
      self, calibration_config: aqt_config.CalibrationConfig) -> tf.Tensor:
    """Compute the upper bound on input tensor values, broadcastable to input."""
    return _bound(
        calibration_config,
        self._config.lp_order,
        tf.zeros(self.stats_shape, dtype=tf.float32),
        self._sum_of_ones.read_value(),
        self._max_of_abs_vals,
        self._sum_of_l1_vals.read_value(),
        self._sum_of_lp_vals.read_value(),
        self.divide,
    )

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


class TensorQuantizerBase:
  """Maintains state associated with the quantization of an input tensor.

  A TensorQuantizerBase provides common functionalities for both static and
  dynamic quantization, recording the last time of the most recent update to
  this `TensorQuantizerBase` class to be able to switch between `tensor_configs`
  in a given `AqtScheduleConfig`.

  TensorQuantizer assumes the supplied `event_count` strictly monotonically
  increases and starts out strictly greater than `tf.int64.min`.

  Uses the `get_variable` provided in the constructor to create variables.

  Attributes:
    data_shape: the shape of the tensor this quantizes. These dimensions may be
      `None`, but then they must correspond to shared stats axes in
      `config.stats_config` when quantization is static. In this case, use of
      quanitzed variable is disallowed as the shapes of the quantization scales
      are not fully defined.
    config: A training quantization schedule.
  """

  def __init__(
      self,  #
      data_shape: Iterable[Optional[int]],
      config: aqt_config.AqtScheduleConfig,
      get_variable: GetVariable = default_get_variable,
      name: str = 'tensor_quantizer_base',
  ):
    self.data_shape = list(data_shape)
    config.fill_gaps_with_float_config()
    config.validate(self.data_shape)
    self.config = config

    with tf.variable_scope(name):
      # This variable maintains the most recent event count at which this
      # TensorQuantizer was updated. This determines which quantization config
      # from our schedule `self.config` is active, which is required because:
      # (1) at inference time, this determines the quantization config to serve.
      # (2) at training time, the first time a config with a frozen scale is
      # active, we must update the scale, a calculation requiring this variable.
      self._last_update = get_variable('last_update', [], tf.int64,
                                       tf.int64.min)

  def tracked_variables(self) -> Dict[str, tf.Variable]:
    """Returns variables used to track updates."""
    return {
        'last_update': self._last_update,
    }

  def calibration_variables(self) -> Dict[str, tf.Variable]:
    """Returns scale and stats variables used to calibrate tensors."""
    raise NotImplementedError

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

  def _quantization_params(self, train):
    """Returns parameters for AQT quantization routine."""

    @dataclasses.dataclass
    class QParams:
      """Dataclass for quantization parameters."""
      # Tensors used for quantization
      should_quantize: tf.Tensor = tf.constant(0.0)
      clip_bound: tf.Tensor = tf.constant(0.0)
      shift_before: tf.Tensor = tf.constant(0.0)
      shift_after: tf.Tensor = tf.constant(0.0)

      # Tensors used for floating point emulation.
      should_use_small_float: tf.Tensor = tf.constant(0)
      mantissa_bits: tf.Tensor = tf.constant(0)
      min_exp: tf.Tensor = tf.constant(0)
      max_exp: tf.Tensor = tf.constant(0)

      def __iter__(self):
        return (getattr(self, field.name) for field in dataclasses.fields(self))

    def qparams(config: aqt_config.AqtTensorConfig,
                check_active: bool = True) -> QParams:
      """Returns quant parameters for fixed AQT config."""

      params = QParams()
      config_active = is_config_active(config, self._last_update.read_value())
      config_active = not check_active or config_active

      if isinstance(config.quant_config, aqt_config.FloatConfig):
        params.clip_bound = tf.where_v2(config_active, float('inf'), 0.0)
      elif isinstance(config.quant_config, aqt_config.IntQuantConfig):
        config_active = tf.cast(config_active, tf.float32)
        params.should_quantize += config_active

        # TODO(vladf): some serving environments, such as adbrain,
        # automatically rewrite constants from f32 to bf16, which makes the
        # epsilon used in the function below invalid, so that
        # safe_clip_bound == clip_bound (incorrectly). We should solve for
        # that through some configuration.
        params.clip_bound += config_active * aqt_common.safe_clip_bound(
            config.quant_config)
        if config.quant_config.preserve_zero:
          params.shift_before += config_active * 0.5
        else:
          params.shift_after += config_active * 0.5
      elif isinstance(config.quant_config, aqt_config.SmallFloatConfig):
        # Note: In the case of small floats, we can just use get_clip_bound
        # as opposed to safe_clip_bound since we are mapping the clip bound
        # directly to the highest float that can be represented, while in int
        # world we are mapping biggest value onto 127.4999 to get a slightly
        # wider and less biased range.
        config_active = tf.cast(config_active, tf.int32)
        params.should_use_small_float += config_active

        params.clip_bound += (
            tf.cast(config_active, tf.float32) *
            aqt_common.get_clip_bound(config.quant_config))

        params.mantissa_bits += (
            config_active * config.quant_config.mantissa_bits)
        params.min_exp += config_active * config.quant_config.min_exp
        params.max_exp += config_active * config.quant_config.max_exp
      else:
        raise ValueError('config.quant_config must have type '
                         'FloatConfig, IntQuantConfig, or SmallFloatConfig. '
                         f'But instead got {type(config.quant_config)}.')

      return params

    if not train and self.config.inference_config_index is not None:
      params = qparams(
          self.config.tensor_configs[self.config.inference_config_index],
          check_active=False)
    elif self.config.tensor_configs:
      params = QParams(*map(tf.add_n, zip(
          *map(qparams, self.config.tensor_configs))))
    else:
      return QParams()

    params.should_quantize = tf.cast(params.should_quantize, tf.bool)
    params.should_use_small_float = tf.cast(
        params.should_use_small_float, tf.bool)
    return params

  def should_quantize(self, train: bool) -> tf.Tensor:
    """Returns the number of quant configs are active."""
    params = self._quantization_params(train)
    return params.should_quantize

  def _to_quant(
      self,
      x: tf.Tensor,
      train: bool,
      use_stochastic_rounding: bool = False,
  ) -> tf.Tensor:
    """Quantizes x with active quant config, if any, else act as identity.

    Args:
      x: tensor to quantize
      train: whether training
      use_stochastic_rounding: whether to add random numbers in [-0.5, 0.5]
        before integer quantization.

    Returns:
      quantized tensor
    """
    with tf.variable_scope('to_quant'):
      params = self._quantization_params(train)

      def _floor(t: tf.Tensor, use_stochastic_rounding: bool) -> tf.Tensor:
        if use_stochastic_rounding:
          t = t + tf.random.uniform(tf.shape(t), -0.5, 0.5, dtype=t.dtype)
        return tf.math.floor(t)

      def maybe_floor_or_small_float(y):
        # Static check for whether int and small float can coexist in schedule
        if self.config.allow_int_small_float:
          return tf.where_v2(params.should_use_small_float,
                             _emulated_fp(y, params.mantissa_bits,
                                          params.min_exp,
                                          params.max_exp),
                             tf.where_v2(params.should_quantize,
                                         tf.math.floor(y), y))
        else:
          if self.config.quantization_mode() == aqt_config.SmallFloatConfig:
            return tf.where_v2(
                params.should_use_small_float,
                _emulated_fp(y, params.mantissa_bits, params.min_exp,
                             params.max_exp), y)
          else:
            return tf.where_v2(
                params.should_quantize, _floor(y, use_stochastic_rounding), y
            )

      # Note that backprop does not depend directly on the value of _last_update
      # or any_config_active; only scales and constants need to be maintained
      # and there's no branching on ops including the input tensor x. This
      # results in significant memory reduction, see cl/415355150.
      x = tf.clip_by_value(x, -params.clip_bound, params.clip_bound)
      x += params.shift_before
      x = tf.grad_pass_through(maybe_floor_or_small_float)(x)
      x += params.shift_after

    return x

  def _clip_mask(self, x: tf.Tensor, train: bool) -> tf.Tensor:
    """Returns which entries of x are clipped in _to_quant(x)."""
    # TODO(vladf): we should consider re-using the same clip mask
    # computation in _to_quant to derive this clip mask instead of a separate
    # method.
    with tf.variable_scope('clip_mask'):
      _, clip_bound, _, _, _, _, _, _ = self._quantization_params(train)
      return tf.abs(x) > clip_bound

  def _should_scale(self, train: bool) -> tf.Tensor:
    """Returns True if any non-float quant config is active."""
    if not train and self.config.inference_config_index is not None:
      inference_config = self.config.tensor_configs[
          self.config.inference_config_index
      ]
      should_scale = tf.constant(
          not isinstance(inference_config.quant_config, aqt_config.FloatConfig)
      )
    else:
      should_scale = tf.constant(False)
      for config in self.config.tensor_configs:
        if isinstance(config.quant_config, aqt_config.FloatConfig):
          continue

        config_active = is_config_active(config, self._last_update)
        should_scale |= config_active
    return should_scale

  def _maybe_fallback_to_ones(
      self,
      should_scale: tf.Tensor,
      scale: tf.Tensor,
      inv_scale: tf.Tensor,
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Fallback to ones if should not scale."""
    # TODO(lew): We can simplify the scopes to just 'compute scales'
    with tf.variable_scope('to_quant'):
      scale = tf.where_v2(should_scale, scale, tf.ones_like(scale))  #
    with tf.variable_scope('from_quant'):
      inv_scale = tf.where_v2(  #
          should_scale, inv_scale, tf.ones_like(inv_scale)
      )
    return scale, inv_scale


class TensorQuantizer(TensorQuantizerBase):
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
    data_shape: the shape of the tensor this quantizes. These dimensions may be
      `None`, but then they must correspond to shared stats axes in
      `config.stats_config`. In this case, use of quanitzed variable is
      disallowed.
    config: A training quantization schedule.
    quantized_variable: If `config.use_quantized_variable`, then the quantized
      version of the most recent `update()` tensor is saved to this member. This
      is helpful in inference settings, where users may be interested in only
      saving quantized versions of the weights to reduce storage consumption or
      avoid quantization of floating point weights at inference time.
  """

  def __init__(
      self,  #
      data_shape: Iterable[Optional[int]],
      config: aqt_config.AqtScheduleConfig,
      get_variable: GetVariable = default_get_variable,
      name: str = 'tensor_quantizer',
  ):
    super().__init__(
        data_shape=data_shape,
        config=config,
        get_variable=get_variable,
        name=name,
    )

    with tf.variable_scope(name):
      self._stats = Stats(
          data_shape=self.data_shape,  #
          config=self.config.stats_config,
          get_variable=get_variable,
      )

      # We intentionally initialize scale to zero to fail loudly if someone uses
      # a parameter such as scale without properly update()-ing it.
      self._scale = get_variable(
          'scale', self._stats.stats_shape, tf.float32, 0
      )
      # Save the inverse scale so that we don't recompute it at inference time.
      self._inv_scale = get_variable(
          'inv_scale', self._stats.stats_shape, tf.float32, 0
      )

      # Variable to save or read quantized tensors to, if the config says so.
      if self.config.use_quantized_variable:
        self.quantized_variable = get_variable(
            'quantized_variable', self.data_shape, tf.int8, 0
        )

  def tracked_variables(self) -> Dict[str, tf.Variable]:
    """Returns variables used to track updates and calibration variables."""
    variables = super().tracked_variables()
    variables.update(self.calibration_variables())
    if self.config.use_quantized_variable:
      variables['quantized_variable'] = self.quantized_variable
    return variables

  def calibration_variables(self) -> Dict[str, tf.Variable]:
    """Returns scale and stats variables used to calibrate tensors."""
    return {
        'scale': self._scale,
        'inv_scale': self._inv_scale,
        **self._stats.calibration_variables(),
    }

  def _fresh_scale(
      self, config: aqt_config.AqtTensorConfig
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Returns new scale, inverse scale for a given config and stats, if any."""
    if isinstance(config.quant_config, aqt_config.FloatConfig):
      # We shouldn't update the scale if the given config contains FloatConfig
      # and no emulation;
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
      if isinstance(config.quant_config, aqt_config.FloatConfig):
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

  def update(
      self,  #
      sample: Optional[tf.Tensor],
      weight: Optional[tf.Tensor],
      event_count: tf.Tensor,
  ) -> tf.Operation:
    """Op to update statistics, scale, and quantized variable.

    Args:
      sample: an observation of the tensor to quantize. If None, only update the
        state variables without updating the stats
      weight: the weight of each observation for updating statistics.
      event_count: the event count this observation corresponds to; this
        determines the scale that's active for any quantized methods referencing
        this `TensorQuantizer`.

    Returns:
      A tensorflow operation corresponding to the updates to internal variables
      for capturing this observation of `sample`.
    """
    dependencies = []
    if sample is not None:
      # update the stats if a sample is passed, else only update state
      dependencies.append(self._stats.update(sample, weight))
    with tf.control_dependencies(dependencies):

      def case_fn(config):
        return self._update_state_config(config, sample, event_count)

      return self._config_case(case_fn, event_count).op

  def _update_state_config(
      self,  #
      config: aqt_config.AqtTensorConfig,
      sample: tf.Tensor,
      event_count: tf.Tensor,
  ) -> tf.Tensor:
    """Returns tensor with dependency on updates for state variables."""
    updates = []

    # Ensure we read _last_update before we update it.
    last_update = self._last_update.read_value()
    with tf.control_dependencies([last_update]):
      should_update_scale = _should_update_scale(
          config, prev_event_count=last_update, new_event_count=event_count  #
      )
      updates.append(self._last_update.assign(event_count))

    updated_scale, updated_inv_scale = tf.cond(
        should_update_scale,  #
        lambda: self._fresh_scale(config),
        lambda: (self._scale.read_value(), self._inv_scale.read_value()),
    )
    updates.append(self._scale.assign(updated_scale))
    updates.append(self._inv_scale.assign(updated_inv_scale))

    if (
        self.config.use_quantized_variable
        and isinstance(config.quant_config, aqt_config.IntQuantConfig)
        and config.quant_config.compatible_with_int8()
    ):
      with tf.control_dependencies(updates):
        scale, _ = self._get_quant_scale(train=True)
        sample = scale * sample
        updated_quantized_variable = tf.cast(
            self._to_quant(sample, train=True), self.quantized_variable.dtype
        )
        updates.append(
            self.quantized_variable.assign(updated_quantized_variable)
        )

    with tf.control_dependencies(updates):
      return tf.constant(0)

  def _get_quant_scale(self, train: bool) -> tf.Tensor:
    """Returns scales to quantize/dequantize the active quant config, if any, else ones."""
    should_scale = self._should_scale(train)

    return self._maybe_fallback_to_ones(
        should_scale,
        self._scale,
        self._inv_scale,
    )


def validate_dynamic(config: aqt_config.AqtScheduleConfig) -> None:
  """Validates the config conforms with dynamic quantization."""
  if config.stats_config.ema_update_count != 1:
    raise aqt_config.ConfigError(
        'ema_update_count={config.stats_config.ema_update_count} must be 1 '
        'for dynamic quantization.'
    )
  if config.use_quantized_variable:
    raise aqt_config.ConfigError(
        'dynamic quantization  does not memorized the quantized variable as '
        'it is history-independent.'
    )
  for tensor_config in config.tensor_configs:
    # When a tensor config is not FloatConfig, the quantizer may use
    # non-trivial quantization scales and dynamic quantization should not use
    # scales in such cases.
    if (
        isinstance(tensor_config, aqt_config.FloatConfig)
        and tensor_config.freeze_scale_at_begin
    ):
      raise aqt_config.ConfigError(
          'Dynamic quantization should not freeze_scale_at_begin for non-float '
          'config but got {tensor_config}.'
      )


class DynamicTensorQuantizer(TensorQuantizerBase):
  """Maintains state associated with the dynamic quantization of an input tensor.

  A DynamicTensorQuantizer quantize a input tensor dynamically. It does not
  maintain any statistics, scales, or quantized variables for the forward pass.

  DynamicTensorQuantizer inherits from TensorQuantizerBase that uses the
  `get_variable` provided in the constructor to create a variable to memorize
  the most recent event_count and provides methods to switch between
  `tensor_configs` in a given `AqtScheduleConfig`

  DynamicTensorQuantizer assumes the supplied `event_count` strictly
  monotonically increases and starts out strictly greater than `tf.int64.min`.


  Attributes:
    data_shape: the shape of the tensor this quantizes. These dimensions may be
      `None`.
    config: A training quantization schedule.
  """

  def __init__(
      self,  #
      data_shape: Iterable[Optional[int]],
      config: aqt_config.AqtScheduleConfig,
      get_variable: GetVariable = default_get_variable,
      name: str = 'dynamic_tensor_quantizer',
  ):
    validate_dynamic(config)
    super().__init__(
        data_shape=data_shape,
        config=config,
        get_variable=get_variable,
        name=name,
    )

  def calibration_variables(self) -> Dict[str, tf.Variable]:
    """Returns empty dict as dynamic quantization does not store calibration variables."""
    return {}

  def dynamic_clip_range(
      self,
      sample: tf.Tensor,
      weight: Optional[tf.Tensor],
      event_count: tf.Tensor,
      train: bool,
  ) -> tf.Tensor:
    """Returns the tensor clip range or zeros if no int config is active."""
    _, inv_scale = self._get_dynamic_quant_scale(
        sample, weight, event_count, train
    )

    def case_fn(config: aqt_config.AqtTensorConfig) -> tf.Tensor:
      if isinstance(config.quant_config, aqt_config.FloatConfig):
        return tf.zeros_like(inv_scale, dtype=inv_scale.dtype)

      # We return the range derived from the inverse scale, rather than
      # from the stats themselves, to respect freezing settings and
      # report the clipping range that's actually used at all times.
      #
      # The counterfactual clipping range that would have been used
      # if we didn't freeze the scale can be re-derived from the current
      # stats values, which are updated regardless of freezing.
      clip_bound = aqt_common.get_clip_bound(config.quant_config)
      return inv_scale * clip_bound

    return self._config_case(case_fn, self._last_update.read_value())

  def _fresh_dynamic_scale(
      self,
      tensor_config: aqt_config.AqtTensorConfig,
      sample: tf.Tensor,
      weight: tf.Tensor,
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Returns new scale, inverse scale for a given config and stats, if any."""
    if isinstance(tensor_config.quant_config, aqt_config.FloatConfig):
      # We shouldn't return the scale if the given config contains FloatConfig
      # and no emulation;
      # fill with a poison value if we get into this situation.
      nan = _init_dynamic_stats(
          self.config.stats_config, sample, init_value=float('nan'))
      return nan, nan

    x_bound = _dynamic_bound(
        self.config.stats_config,
        tensor_config.calibration_config,
        sample,
        weight,
    )

    clip_bound = aqt_common.get_clip_bound(tensor_config.quant_config)

    new_scale = clip_bound / x_bound
    inv_scale = x_bound / clip_bound
    return new_scale, inv_scale

  def _get_dynamic_quant_scale(
      self,
      sample: tf.Tensor,
      weight: Optional[tf.Tensor],
      event_count: tf.Tensor,
      train: bool,
  ) -> tf.Tensor:
    """Returns scales to quantize/dequantize the active quant config, if any, else ones."""

    # We intentionally initialize scale to zero to fail loudly if someone uses
    # a parameter such as scale without properly update()-ing it.
    zeros = _init_dynamic_stats(self.config.stats_config, sample, init_value=0)
    def case_fn(config):
      # only need to update the event_count for dynamic quantizer
      updates = []

      # Ensure we read _last_update before we update it.
      last_update = self._last_update.read_value()
      with tf.control_dependencies([last_update]):
        should_update_scale = _should_update_scale(
            config, prev_event_count=last_update, new_event_count=event_count  #
        )
        updates.append(self._last_update.assign(event_count))

      with tf.control_dependencies(updates):
        scale, inv_scale = tf.cond(
            should_update_scale,  #
            lambda: self._fresh_dynamic_scale(config, sample, weight),
            lambda: (zeros, zeros),
        )
        return scale, inv_scale

    scale, inv_scale = self._config_case(case_fn, event_count)

    # make sure the scale is updated
    with tf.control_dependencies([scale.op, inv_scale.op]):
      should_scale = self._should_scale(train)

    return self._maybe_fallback_to_ones(should_scale, scale, inv_scale)
