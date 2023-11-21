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
"""Common utility functions for AQT ops.
"""

from typing import Callable, Dict, List, Optional

from aqt.common import aqt_config
from aqt.tensorflow import aqt_tensor
import tensorflow.compat.v1 as tf

# TODO(b/220181240): Remove the pylint disable below and avoid using protected
# methods.
# We repeatedly use protected methods from classes defined in other modules to
# avoid exporting them as part of the public API.
# pylint: disable=protected-access

TensorQuantizer = aqt_tensor.TensorQuantizer
DynamicTensorQuantizer = aqt_tensor.DynamicTensorQuantizer


def _possibly_use_quantized_variable(
    quantizer: TensorQuantizer, x: tf.Tensor, train: bool  #
) -> tf.Tensor:
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
    x: lhs or rhs of matmul or einsum.
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


def diagnostics(
    op,
    lhs: tf.Tensor,
    rhs: tf.Tensor,
    grad: Optional[tf.Tensor] = None,
) -> Dict[str, tf.Tensor]:
  """Returns a dictionary from keys to diagnostic tensors.

  Args:
    op: A quantized binary op.
    lhs: lhs argument to op.Apply, used for deriving diangostics relative to
      a given input.
    rhs: as above, but for rhs
    grad: If specified, the gradient for deriving diagnostics.

  Returns:
    A dictionary with various quantization-related diagnostics,
    whose string keys are prefixed by op.name/op.{lhs,rhs}_name.
  """
  d = {}
  quantizers = [
      (op.lhs_name, op.lhs_quantizer, lhs),
      (op.rhs_name, op.rhs_quantizer, rhs),
  ]
  if grad is not None:
    assert op.grad_quantizer is not None, (
        'If grad is given, then grad_quantizer must be defined.')
    quantizers.append((op.grad_name, op.grad_quantizer, grad))
  for prefix, quantizer, argument in quantizers:
    clipped_proportion = tf.cast(tf.abs(argument) > quantizer.clip_range(),
                                 tf.float32)
    prefix = f'{op.name}/{prefix}'
    d[f'{prefix}/clipped_proportion'] = tf.math.reduce_mean(
        clipped_proportion)
    d[f'{prefix}/clip'] = quantizer.clip_range()
    d[f'{prefix}/event_count'] = quantizer._last_update
    for name, var in quantizer.calibration_variables().items():
      d[f'{prefix}/{name}'] = var
  return d


def _dense_op_case(
    lhs_quantizer: TensorQuantizer | DynamicTensorQuantizer,
    rhs_quantizer: TensorQuantizer,
    default_dense_op: Callable[[], tf.Tensor],
    int8_dense_op: Callable[[], tf.Tensor],
    train: bool,
    skip_cond_if_all_int_quant: bool = False,
) -> tf.Tensor:
  """Switch over the dense compute based on event count and configs.

  The `TensorQuantizer`s for each argument are provided to supply the
  configurations for each input tensor. Also, if indicated by constituent
  configs, this uses the quantized variables in each tensor quantizer rather
  than the inputs themselves. Regardless, gradients are preserved as if this was
  a matmul over the original `(lhs, rhs)` inputs.

  Args:
    lhs_quantizer: TensorQuantizer for lhs.
    rhs_quantizer: TensorQuantizer for rhs.
    default_dense_op: function for the float dense op.
    int8_dense_op: function for the int8 dense op.
    train: If false and `TQ.use_quantized_variable` is True, then use the
      quantized variable, instead of input tensors, for the respective input
      tensor.
    skip_cond_if_all_int_quant: If all quant configs are IntQuantConfig, skip
      tf.cond.

  Returns:
    The `tf.Tensor` from the resulting quantized dense op.
  """

  lhs_configs = lhs_quantizer.config.tensor_configs
  rhs_configs = rhs_quantizer.config.tensor_configs

  def is_int8_compatible(
      lhs_config: aqt_config.AqtTensorConfig,
      rhs_config: aqt_config.AqtTensorConfig,
  ) -> bool:
    return (
        isinstance(lhs_config.quant_config, aqt_config.IntQuantConfig)
        and isinstance(rhs_config.quant_config, aqt_config.IntQuantConfig)
        and lhs_config.quant_config.bits <= 8
        and rhs_config.quant_config.bits <= 8
    )

  def all_int8_compatible(
      lhs_configs: List[aqt_config.AqtTensorConfig],
      rhs_configs: List[aqt_config.AqtTensorConfig],
  ) -> bool:
    for lhs_config in lhs_configs:
      for rhs_config in rhs_configs:
        if not is_int8_compatible(lhs_config, rhs_config):
          return False
    return True

  # short circuit for quantization all the way, which is not enabled by default
  # for backward checkpoint compatibility.
  if skip_cond_if_all_int_quant and all_int8_compatible(
      lhs_configs, rhs_configs
  ):
    return int8_dense_op()

  lhs_index = lhs_quantizer.config.inference_config_index
  rhs_index = rhs_quantizer.config.inference_config_index

  if train or lhs_index is None or rhs_index is None:
    should_int8_quantize = tf.constant(False)
    for lhs_config in lhs_configs:
      for rhs_config in rhs_configs:
        # If any of lhs and rhs is FloatConfig, use the default matmul.
        if is_int8_compatible(lhs_config, rhs_config):
          should_int8_quantize |= aqt_tensor.is_config_active(
              lhs_config, lhs_quantizer._last_update
          ) & aqt_tensor.is_config_active(
              rhs_config, rhs_quantizer._last_update
          )
    # Use go/tf-control-flow-v2, which we've noticed fuses better on TPU XLA.
    v2_was_enabled = tf.control_flow_v2_enabled()
    if not v2_was_enabled:
      tf.enable_control_flow_v2()
    cond = tf.cond(should_int8_quantize, int8_dense_op, default_dense_op)
    if not v2_was_enabled:
      tf.disable_control_flow_v2()
  else:
    # In the inference setting, if inference config indices are specified,
    # then manually const-prop the tf.cond to avoid overheads such as loading
    # bf16 weights
    should_int8_quantize = is_int8_compatible(
        lhs_configs[lhs_index], rhs_configs[rhs_index]
    )
    if should_int8_quantize:
      cond = int8_dense_op()
    else:
      cond = default_dense_op()
  return cond
