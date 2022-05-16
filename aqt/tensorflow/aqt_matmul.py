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

from typing import Callable

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
      return mm * (lhs_inv_scale * rhs_inv_scale)
