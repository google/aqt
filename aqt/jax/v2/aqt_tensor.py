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
"""Quantization utils for tensors."""

# Lingo in this file:
#
# - lhs(rhs) - left(right) hand side of a binary operation
# - ca - contraction axes
# - ba - batch axes
# - ra - remaining axes

# pylint: disable=g-explicit-bool-comparison
# pylint: disable=g-explicit-length-test
import typing
from typing import Any, Callable, Optional, Sequence, TypeAlias
from aqt.jax.v2 import utils
import flax.cursor
import flax.struct
import jax
import jax.numpy as jnp
import jax.typing as jax_typing
from typing_extensions import Self  # for python version < 3.11

GradientFn = Callable[..., Any] | None  # None when there is no numerics
_MSG_NO_QVALUE = (
    'QTensor does not have qvalue, but it is asked to access the qvalue.'
    ' Call QTensor.quant() before using the qvalue.'
)

if typing.TYPE_CHECKING:
  # These are needed to avoid static typing complaints for sharding.
  # Concatenating sharding types, e. g. jnp.ndarray | jax.sharding.NamedSharding
  # | jax.sharding.PartitionSpec, will trigger an error to logics where the
  # values are used as jax.array, since they could be sharding.
  ArrayT: TypeAlias = Any
else:
  ArrayT: TypeAlias = jnp.ndarray


@utils.flax_slots_dataclass
class QTensor:
  """Quantized tensor."""

  # Quantized (compressed) representation of tensor.
  # Use dequant() method to "decompress" to the original tensor.
  qvalue: Optional[ArrayT]

  # (scale == None) means that scale is unknown/invalid;
  # Otherwise, check dequant(self) for semantics.
  scale: Optional[list[ArrayT]]

  # Used in dot_general, transposed scales used in post dot_general scaling.
  # The same comments apply as to scale.
  # We currently keep it here because:
  # - we store scale_t in the checkpoint to avoid transposition per inference.
  # - scale_t is used both in backprop of dot_general and in post-scaling.
  #   We avoid transposing scale twice.
  # TODO(lew): Move scale_t from QTensor to some dot-general specific type?
  scale_t: Optional[list[ArrayT]]

  # DType of the tensor before quantized.
  dequant_dtype: Optional[jnp.dtype] = flax.struct.field(
      pytree_node=False, default=None
  )

  def quant(self, x):
    assert self.qvalue is None, 'Already quantized QTensor.'
    assert self.scale is not None, 'Missing scales to be used for quantization.'

    qvalue = x
    for s in self.scale:
      qvalue = qvalue * jax.lax.reciprocal(s)

    # TODO(lew): We should apply numerics here, so that 'quant' function
    # Can be considered a part of API.
    return self.replace(qvalue=qvalue)  # pytype: disable=attribute-error

  def dequant(self) -> jnp.ndarray:
    """Dequantizes the QTensor."""
    assert self.scale is not None, 'Missing scales when dequantizing a QTensor.'
    msg = (
        'QTensor is manually created without setting a dequant_detype. It can'
        ' be used in dot_general, but to dequantize you need to set its dtype.'
    )
    assert self.dequant_dtype is not None, msg
    assert self.qvalue is not None, _MSG_NO_QVALUE
    ret = self.qvalue
    for scale in self.scale:
      ret = ret.astype(self.dequant_dtype) * scale.astype(self.dequant_dtype)  # pytype: disable=attribute-error
    return ret  # pytype: disable=bad-return-type

  def qvalue_astype(self, dtype) -> Self:
    assert self.qvalue is not None, _MSG_NO_QVALUE
    return self.replace(qvalue=self.qvalue.astype(dtype))  # pytype: disable=attribute-error

  def __getitem__(self, idx: jax_typing.ArrayLike) -> Self:
    """Returns the indexed subtensor on the first axis."""
    assert self.scale_t is None, 'scale_t is not supported in __getitem__'
    assert self.qvalue is not None, _MSG_NO_QVALUE
    qvalue = self.qvalue[idx]
    scale = [s[idx] for s in self.scale]
    return QTensor(
        qvalue=qvalue,
        scale=scale,
        scale_t=self.scale_t,
        dequant_dtype=self.dequant_dtype,
    )

  @property
  def ndim(self) -> int:
    assert self.qvalue is not None, _MSG_NO_QVALUE
    return self.qvalue.ndim  # pytype: disable=attribute-error

  @property
  def shape(self) -> Sequence[int]:
    assert self.qvalue is not None, _MSG_NO_QVALUE
    return self.qvalue.shape  # pytype: disable=attribute-error


def zeros(
    shape: Sequence[int], qdtype: jnp.dtype, dequant_dtype: jnp.dtype
) -> QTensor:
  return QTensor(
      qvalue=jnp.zeros(shape, dtype=qdtype),
      scale=[],
      scale_t=[],
      dequant_dtype=dequant_dtype,
  )


def zeros_with_scale(
    shape: Sequence[int],
    calibration_axis: Sequence[int],
    qdtype: jnp.dtype,
    dequant_dtype: jnp.dtype,
) -> QTensor:
  """Initializes a QTensor with empty qvalue along with empty scale value."""
  scale_shape = list(shape)
  for axis in calibration_axis:
    scale_shape[axis] = 1

  # TODO(lew): hardcode dequant_dtype to bf16. This requires updating
  # other libraries to not break their functionality.
  return QTensor(
      jnp.zeros(shape, dtype=qdtype),
      [jnp.ones(scale_shape, dtype=dequant_dtype)],
      None,
      dequant_dtype=dequant_dtype,
  )


def dynamic_slice(
    operand: QTensor,
    start_indices: Sequence[int],
    slice_sizes: Sequence[int],
) -> QTensor:
  """Dynamically slices the value at start_indices using the given shape."""
  msg = 'scale_t is not supported in the dynamic_slice of a QTensor.'
  assert operand.scale_t is None, msg

  def get_sliced_scales(scale):
    msg = 'Slice sizes must have the same length as operand dims.'
    assert scale.ndim == len(slice_sizes), msg
    scale_start_indices = list(start_indices)
    scale_slice_sizes = list(slice_sizes)
    for axis in range(len(scale_slice_sizes)):
      # slice size must be <= operand shape
      # scale_slice_sizes[dim] = min(scale_slice_sizes[dim], scale.shape[dim])
      if scale.shape[axis] == 1:
        scale_start_indices[axis] = 0
        scale_slice_sizes[axis] = 1
      msg = (
          'We do not support window overflow that is supported in'
          ' jax.lax.dynamic_slices. Please email lew@google.com if you think'
          ' this is wrong.'
      )
      assert (
          scale_start_indices[axis] + scale_slice_sizes[axis]
          <= scale.shape[axis]
      ), msg
    return jax.lax.dynamic_slice(scale, scale_start_indices, scale_slice_sizes)

  return QTensor(
      jax.lax.dynamic_slice(operand.qvalue, start_indices, slice_sizes),
      [get_sliced_scales(s) for s in operand.scale],
      None,
      operand.dequant_dtype,
  )


def dynamic_update_slice(
    operand: QTensor, update: QTensor, start_indices: Sequence[int]
) -> QTensor:
  """Updates the value at start_indices with the given QTensor value."""
  # This function only works for a specific case, i.e., updating an entire slice
  # along the calibration axis.
  msg = 'scale_t is not supported in dynamic_update_slice'
  assert operand.scale_t is None, msg
  op_dd = operand.dequant_dtype
  up_dd = update.dequant_dtype
  assert op_dd == up_dd, f'Dequant dtype mismatch: {op_dd} != {up_dd}'
  ndim = operand.qvalue.ndim
  assert update.qvalue.ndim == ndim
  for scale, update_scale in zip(operand.scale, update.scale):
    assert scale.ndim == ndim
    assert update_scale.ndim == ndim
    for axis in range(ndim):
      if scale.shape[axis] == 1:
        calibration_axis_shape = operand.qvalue.shape[axis]
        update_calibration_axis_shape = update.qvalue.shape[axis]
        msg = (
            'Only updating an entire slice along the calibration axis is'
            f' valid. The calibration axis shape is {calibration_axis_shape}'
            f' but the update slice shape is {update_calibration_axis_shape}.'
        )
        assert update.qvalue.shape[axis] == operand.qvalue.shape[axis], msg
        msg = 'Update scale and scale should be calibrated along the same axis.'
        assert update_scale.shape[axis] == 1, msg
      else:
        # individual scale per element always works
        pass

  qvalues = jax.lax.dynamic_update_slice(
      operand.qvalue, update.qvalue, start_indices
  )
  scales = [
      jax.lax.dynamic_update_slice(scale, update_scale, start_indices)
      for scale, update_scale in zip(operand.scale, update.scale)
  ]

  return QTensor(qvalues, scales, None, operand.dequant_dtype)


def update_frame(operand: QTensor, frame: int, update: QTensor) -> QTensor:
  """Updates the value at frame with the given QTensor value."""
  assert operand.ndim == update.ndim + 1
  assert operand.dequant_dtype == update.dequant_dtype, 'Dequant dtype mismatch'

  return QTensor(
      operand.qvalue.at[frame].set(update.qvalue),
      [
          target_scale.at[frame].set(update_scale)
          for target_scale, update_scale in zip(operand.scale, update.scale)
      ],
      None,
      operand.dequant_dtype,
  )
