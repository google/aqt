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

from typing import Dict, Optional

from aqt.tensorflow import aqt_tensor
import tensorflow.compat.v1 as tf

# TODO(b/220181240): Remove the pylint disable below and avoid using protected
# methods.
# We repeatedly use protected methods from classes defined in other modules to
# avoid exporting them as part of the public API.
# pylint: disable=protected-access


def _possibly_use_quantized_variable(
    quantizer: aqt_tensor.TensorQuantizer,  #
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
