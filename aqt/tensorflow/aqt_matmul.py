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

"""Quantized matrix multiplication.

The matmul operation is a form of tensor product applied to two arguments
`(a, b)` which contracts the penultimate axis of `a` with the ultimate axis
of `b`.

For details on quantized operations and common configuration arguments, see
`aqt_ops`.
"""

from typing import Callable, Dict, Iterable, Optional

from aqt.common import aqt_config
from aqt.tensorflow import aqt_tensor
import tensorflow.compat.v1 as tf


# TODO(b/220181240): Remove the pylint disable below and avoid using protected
# methods.
# We repeatedly use protected methods from classes defined in other modules to
# avoid exporting them as part of the public API.
# pylint: disable=protected-access


MatmulFn = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]


def _possibly_use_quantized_variable(
    quantizer: aqt_tensor.TensorQuantizer,
    x: tf.Tensor,
    train: bool) -> tf.Tensor:
  """Returns quantized variable if not training and TQ.use_quantized_variable, casted to x.dtype.

  Given an input tensor and its tensor quantizer, here we enforce to use the
  quantized variable stored in the tensor quantizer as long as
  TQ.use_quantized_variable is true and it is in inference, no matter if
  FloatConfig is specified or not.

  The semantics of FloatConfig which is meant not to use quantized variables
  during inference should be respected by requiring users to specify
  TensorQuantizer.use_quantized_variable=False. See more details at
  b/219040448.

  Args:
    quantizer: TensorQuantizer for the input tensor x.
    x: lhs or rhs of matmul.
    train: Indicates if in training or not.

  Returns:
    The input tensor x or its quantized one.
  """
  if quantizer.config.use_quantized_variable and not train:
    qx = quantizer.quantized_variable
    if qx.dtype != x.dtype:
      qx = tf.cast(qx, x.dtype)
    return qx
  return x


def default_matmul(
    lhs_quantizer: aqt_tensor.TensorQuantizer,  #
    rhs_quantizer: aqt_tensor.TensorQuantizer,
    lhs: tf.Tensor,
    rhs: tf.Tensor,
    train: bool) -> tf.Tensor:
  """Perform tf.matmul with input tensors of float32 type."""
  lhs = _possibly_use_quantized_variable(lhs_quantizer, lhs, train)
  rhs = _possibly_use_quantized_variable(rhs_quantizer, rhs, train)

  return tf.matmul(lhs, rhs)


def int8_matmul(
    lhs_quantizer: aqt_tensor.TensorQuantizer,  #
    rhs_quantizer: aqt_tensor.TensorQuantizer,
    lhs: tf.Tensor,
    rhs: tf.Tensor,
    train: bool) -> tf.Tensor:
  """Perform integral matmul after i8 cast while preserving gradients.

  1. If {lhs,rhs}_quantizer indicates the currently-active configuration
     should use saved quantized variables, then uses them. Otherwise,
     casts lhs/rhs to tf.int8.
  2. Performs native int8 matmul.
  3. Casts int32 result to f32.

  Despite the effictive clipping, it has a gradient of a float matmul.

  Args:
    lhs_quantizer: TensorQuantizer for lhs.
    rhs_quantizer: TensorQuantizer for rhs.
    lhs: left hand side of the matmul, as a float.
    rhs: right hand side of the matmul, as a float.
    train: If false and `TQ.use_quantized_variable` is True, then use the
      quantized variable, instead of input tensors, for the respective input
      tensor.

  Returns:
    The result of the integer matmul.
  """

  def grad_fn(dy, variables=None):
    # We could manually write the backward pass of matmul with respect to the
    # inputs lhs, rhs by deriving an equivalent expression with matrix
    # calculus. However, with higher-rank tensors, the axes handling becomes
    # cumbersome. Since the backward pass is in floating point right now anyway,
    # just generate a temporary forward matmul op (the one we quantized in the
    # forward pass) and ask autograd to derive the backwards direction itself.
    mm = tf.matmul(lhs, rhs)
    grads = tf.gradients(
        ys=[mm],
        xs=[lhs, rhs],
        # Stop grads in case inputs are aliased.
        stop_gradients=[lhs, rhs],
        grad_ys=[dy])

    # The variables used by fwd() are quantized, so they have no gradients.
    if variables:
      return grads, [None] * len(variables)
    return grads

  @tf.custom_gradient
  def fwd(arg_lhs, arg_rhs):
    # Wrap a custom-gradient op around the cast to propagate gradients,
    # since casting stops the gradient.
    int_lhs = tf.cast(arg_lhs, tf.int8)
    int_rhs = tf.cast(arg_rhs, tf.int8)

    int_lhs = _possibly_use_quantized_variable(lhs_quantizer, int_lhs, train)
    int_rhs = _possibly_use_quantized_variable(rhs_quantizer, int_rhs, train)

    imm = tf.matmul(int_lhs, int_rhs, output_type=tf.int32)
    mm = tf.cast(imm, tf.float32)
    return mm, grad_fn

  return fwd(lhs, rhs)


def _matmul_case(lhs_quantizer, rhs_quantizer, lhs, rhs, train):
  """Switch over matmuls based on event count and configs.

  The `TensorQuantizer`s for each argument are provided to supply the
  configurations for each input tensor. Also, if indicated by constituent
  configs, this uses the quantized variables in each tensor quantizer rather
  than the inputs themselves. Regardless, gradients are preserved as if this was
  a matmul over the original `(lhs, rhs)` inputs.

  Args:
    lhs_quantizer: TensorQuantizer for lhs.
    rhs_quantizer: TensorQuantizer for rhs.
    lhs: float input for lhs.
    rhs: float input for rhs.
    train: If false and `TQ.use_quantized_variable` is True, then use the
    quantized variable, instead of input tensors, for the respective input
    tensor.

  Returns:
    The `tf.Tensor` from the resulting quantized matmul.
  """

  lhs_configs = lhs_quantizer.config.tensor_configs
  rhs_configs = rhs_quantizer.config.tensor_configs

  def is_int8_compatible(lhs_config, rhs_config):
    return (isinstance(lhs_config.quant_config, aqt_config.IntQuantConfig) and
            isinstance(rhs_config.quant_config, aqt_config.IntQuantConfig) and
            lhs_config.quant_config.bits <= 8 and
            rhs_config.quant_config.bits <= 8)

  lhs_index = lhs_quantizer.config.inference_config_index
  rhs_index = rhs_quantizer.config.inference_config_index

  if train or lhs_index is None or rhs_index is None:
    should_int8_quantize = tf.constant(False)
    for lhs_config in lhs_configs:
      for rhs_config in rhs_configs:
        # If any of lhs and rhs is FloatConfig, use the default matmul.
        if is_int8_compatible(lhs_config, rhs_config):
          should_int8_quantize |= (
              aqt_tensor.is_config_active(lhs_config,
                                          lhs_quantizer._last_update)
              & aqt_tensor.is_config_active(rhs_config,
                                            rhs_quantizer._last_update))
  else:
    should_int8_quantize = tf.constant(
        is_int8_compatible(lhs_configs[lhs_index], rhs_configs[rhs_index]))

  # Use go/tf-control-flow-v2, which we've noticed fuses better on TPU XLA.
  v2_was_enabled = tf.control_flow_v2_enabled()
  if not v2_was_enabled:
    tf.enable_control_flow_v2()
  cond = tf.cond(
      should_int8_quantize,
      lambda: int8_matmul(lhs_quantizer, rhs_quantizer, lhs, rhs, train),
      lambda: default_matmul(lhs_quantizer, rhs_quantizer, lhs, rhs, train))
  if not v2_was_enabled:
    tf.disable_control_flow_v2()
  return cond


def _validate_inputs(
    lhs_quantizer: aqt_tensor.TensorQuantizer,  #
    rhs_quantizer: aqt_tensor.TensorQuantizer,
):
  """Validates configs and inputs for matmul."""

  if len(lhs_quantizer.data_shape) != 2:
    raise aqt_config.ConfigError(
        f'lhs data shape ({lhs_quantizer.data_shape}) not rank 2')

  if len(rhs_quantizer.data_shape) != 2:
    raise aqt_config.ConfigError(
        f'rhs data shape ({rhs_quantizer.data_shape}) not rank 2')

  lhs_config = lhs_quantizer.config
  rhs_config = rhs_quantizer.config

  aqt_config._validate_alignment(
      'lhs_config',  #
      lhs_config.tensor_configs,
      'rhs_config',
      rhs_config.tensor_configs)

  if 1 not in lhs_config.stats_config.share_stats_axes:
    raise aqt_config.ConfigError(
        f'expected lhs matmul contraction axis (1) to be in '
        f'share_stats_axes={lhs_config.stats_config.share_stats_axes}')
  if 0 not in rhs_config.stats_config.share_stats_axes:
    raise aqt_config.ConfigError(
        f'expected rhs matmul contraction axis (0) to be in '
        f'share_stats_axes={rhs_config.stats_config.share_stats_axes}')


def matmul(
    lhs_quantizer: aqt_tensor.TensorQuantizer,  #
    lhs: tf.Tensor,
    rhs_quantizer: aqt_tensor.TensorQuantizer,
    rhs: tf.Tensor,
    train: bool = True
) -> tf.Tensor:
  """Quantized tf.matmul.

  Gradients are propagated using the straight-through estimator [1] to
  `lhs` and `rhs` arguments. Note that this is a pure function, with
  quantization determined by the argument quantizers, which must be
  updated separately.

  If not `lhs_quantizer.train` (which should be equal to that of RHS)
  and the quantizers have cached quantized variables, then those are
  used instead, if the quantization config indicates as much.

  Args:
    lhs_quantizer: the tensor quantizer for lhs
    lhs: left-hand side of the matmul
    rhs_quantizer: the tensor quantizer for rhs
    rhs: right-hand side of the matmul
    train: If false and `use_quantized_variable` in lhs_quantizer or
      rhs_quantizer, then this indicates `aqt_matmul` should use the quantized
      variable with the latest quantized, memorized from the most recent
      `TensorQuantizer.update()` in quantized operations rather than the float
      tensor input `lhs` or `rhs` provided to those operations at inference
      time.

  Returns:
    Approximate Matmul result.

  Raises:
    aqt_config.ConfigError: if the configurations for the two quantizers
      are incompatible or unaligned (endpoints for start/stop event counts
      must be the same).

  [1]: https://arxiv.org/abs/1308.3432
  """
  _validate_inputs(lhs_quantizer, rhs_quantizer)

  def fwd(lhs, rhs):
    with tf.name_scope('AqtMatMul'):
      with tf.name_scope('to_quant'):
        with tf.name_scope('lhs'):
          lhs = lhs_quantizer._to_quant(lhs, train)
        with tf.name_scope('rhs'):
          rhs = rhs_quantizer._to_quant(rhs, train)

      with tf.name_scope('matmul'):
        mm = _matmul_case(lhs_quantizer, rhs_quantizer, lhs, rhs, train)

      with tf.name_scope('from_quant'):
        with tf.name_scope('lhs'):
          lhs_inv_scale = lhs_quantizer._from_quant_scale(train)
        with tf.name_scope('rhs'):
          rhs_inv_scale = rhs_quantizer._from_quant_scale(train)

      with tf.name_scope('inv_scale'):
        # TODO(b/236024344): consider alternative multiply associations here.
        inv_scale = (lhs_inv_scale * rhs_inv_scale)
        out = mm * inv_scale

    return out

  @tf.custom_gradient
  def qmatmul(lhs, rhs):

    out = fwd(lhs, rhs)

    def bwd(grad):
      # Make sure to build all backprop results after fprop is computed.
      # Since we rely on variables being updated, this is important for
      # consistency. For instance, the forward pass might be computed under
      # a user-added control dependency from Matmul.update_{lhs,rhs}; the
      # backward pass should also share this dependency transitively.
      with tf.control_dependencies([out]):

        # Note the differences between autograd and what we
        # have implemented below:
        #
        # (1) We re-quantize lhs, rhs to save memory between
        #     fprop and bprop, this is manual rematerialization.
        # (2) Each bprop matmul can avoid a multiplication
        #     by an scale and inverse scale of an fprop argument
        #     due to cancellation.

        with tf.name_scope('BwdAqtMatMul'):

          with tf.name_scope('from_quant'):
            with tf.name_scope('lhs'):
              lhs_inv_scale = lhs_quantizer._from_quant_scale(train)
            with tf.name_scope('rhs'):
              rhs_inv_scale = rhs_quantizer._from_quant_scale(train)

          with tf.name_scope('lhs'):
            qrhs = rhs_quantizer._to_quant(rhs, train)
            lhs_bwd = tf.matmul(grad * rhs_inv_scale, tf.transpose(qrhs))
            lhs_bwd = tf.where_v2(
                lhs_quantizer._clip_mask(lhs, train), 0.0, lhs_bwd)

          with tf.name_scope('rhs'):
            qlhs = lhs_quantizer._to_quant(lhs, train)
            rhs_bwd = tf.matmul(tf.transpose(qlhs), grad * lhs_inv_scale)
            rhs_bwd = tf.where_v2(
                rhs_quantizer._clip_mask(rhs, train), 0.0, rhs_bwd)

        return [lhs_bwd, rhs_bwd]

    return out, bwd

  return qmatmul(lhs, rhs)


class Matmul:
  """Encapsulates a quantized matmul op and calibration state."""

  def __init__(self,
               config: aqt_config.AqtMatmulConfig,
               lhs_shape: Iterable[Optional[int]],
               rhs_shape: Iterable[Optional[int]],
               name: str = 'aqt',
               lhs_name: str = 'lhs',
               rhs_name: str = 'rhs'):
    """Creates a Matmul instance.

    This encapsulates the state necessary for quantizing both
    of the matmul arguments.

    Args:
      config: an AqtMatmulConfig
      lhs_shape: shape of the lhs tensor to multiply
      rhs_shape: shape of the rhs tensor to multiply
      name: variable scope for all variables to be created.
      lhs_name: scope for left hand side variables only.
      rhs_name: scope for right hand side variables only.
    """
    self.name = name
    self.lhs_name = lhs_name
    self.rhs_name = rhs_name
    with tf.variable_scope(name):
      self.lhs_quantizer = aqt_tensor.TensorQuantizer(
          lhs_shape, config.lhs, name=lhs_name)
      self.rhs_quantizer = aqt_tensor.TensorQuantizer(
          rhs_shape, config.rhs, name=rhs_name)

    _validate_inputs(self.lhs_quantizer, self.rhs_quantizer)

  def update_lhs(self, x: tf.Tensor, weights: tf.Tensor,
                 event_count: tf.Tensor) -> tf.Operation:
    """Updates variables for an observation of the lhs.

    Updating variables for an argument updates the statistics
    for a new input to account for incremental observations of
    a tensor's entries' magnitudes.

    This also updates the scales, if they were set according to
    a previous calibration config and now we've moved on to
    a new event associated with a different calibration configuration
    in the schedule.

    Args:
      x: a tensor for the observation of an lhs input
      weights: a weight matrix broadcastable to x, representing how much weight
        the corresponding axes should have on statistics associated with
        quantizing that dimension.
      event_count: the event of the observation

    Returns:
      The tf.Operation corresponding to the update
    """
    return self.lhs_quantizer.update(x, weights, event_count)

  def update_rhs(self, x: tf.Tensor, weights: tf.Tensor,
                 event_count: tf.Tensor) -> tf.Operation:
    """Computes analogue of update_lhs, but for rhs."""
    return self.rhs_quantizer.update(x, weights, event_count)

  def apply(self,
            lhs: tf.Tensor,
            rhs: tf.Tensor,
            train: bool = True) -> tf.Tensor:
    """Generates a pure quantized matmul op.

    Make sure that `apply` is called within the context of any updates
    to statistics used for calibration you'd like to happen before the
    op.

    Args:
      lhs: a float32 tensor for the left hand side
      rhs: a float32 tensor for the right hand side
      train: whether to generate the training or serving graph

    Returns:
      A tf.Tensor generated from possibly quantizing lhs and rhs
      with clip bounds derived from the current quantizer statistics.
    """

    return matmul(self.lhs_quantizer, lhs, self.rhs_quantizer, rhs, train=train)

  def diagnostics(self, lhs: tf.Tensor, rhs: tf.Tensor) -> Dict[str, tf.Tensor]:
    """Returns a dictionary from keys to diagnostic tensors.

    Args:
      lhs: lhs argument to self.Apply, used for derving diangostics relative to
        a given input.
      rhs: as above, but for rhs

    Returns:
      A dictionary with various quantization-related diagnostics,
      whose string keys are prefixed by self.name/self.{lhs,rhs}_name.
    """
    d = {}
    for prefix, quantizer, argument in [
        (self.lhs_name, self.lhs_quantizer, lhs),
        (self.rhs_name, self.rhs_quantizer, rhs)
    ]:
      clipped_proportion = tf.cast(argument > quantizer.clip_range(),
                                   tf.float32)
      prefix = f'{self.name}/{prefix}'
      d[f'{prefix}/clipped_proportion'] = tf.math.reduce_mean(
          clipped_proportion)
      d[f'{prefix}/clip'] = quantizer.clip_range()
      for name, var in quantizer.calibration_variables().items():
        d[f'{prefix}/{name}'] = var
    return d
