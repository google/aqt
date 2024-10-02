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
from typing import Any, Callable, Sequence, TypeAlias
from aqt.jax.v2 import tiled_dot_general
from aqt.jax.v2 import utils
import flax.cursor
import flax.struct
import jax
import jax.numpy as jnp
import jax.typing as jax_typing
from typing_extensions import Self  # for python version < 3.11

GradientFn = None | Callable[..., Any]  # None when there is no numerics
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

TilingState = tiled_dot_general.TilingState


@utils.flax_slots_kw_only_dataclass
class QTensor:
  """Quantized tensor."""

  # Quantized (compressed) representation of tensor.
  # Use dequant() method to "decompress" to the original tensor.
  qvalue: None | ArrayT

  # (scale == None) means that scale is unknown/invalid;
  # Otherwise, check dequant(self) for semantics.
  scale: None | list[ArrayT]

  # Used in dot_general, transposed scales used in post dot_general scaling.
  # The same comments apply as to scale.
  # We currently keep it here because:
  # - we store scale_t in the checkpoint to avoid transposition per inference.
  # - scale_t is used both in backprop of dot_general and in post-scaling.
  #   We avoid transposing scale twice.
  # TODO(lew): Move scale_t from QTensor to some dot-general specific type?
  scale_t: None | list[ArrayT]

  # len(bias) == 0 means bias should not be applied.
  # Quantization and dequantization are defined such that:
  #   quant(x) = (x + b) / s = (x + b[0] + b[1] + ...) / s[0] / s[1] / ...
  # dequant(q) = (q * s) - b = (q * s[0] * s[1] * ...) - b[0] - b[1] - ...
  bias: list[ArrayT]

  # DType of the tensor before quantized.
  # NOTE: AQT Users should use the public property, dtype, instead.
  dequant_dtype: None | jnp.dtype = flax.struct.field(
      pytree_node=False, default=None
  )

  # Tiling state of the tensor.
  tiling_state: None | TilingState = flax.struct.field(
      pytree_node=False, default=None
  )

  @property
  def dtype(self) -> None | jnp.dtype:
    # As some dequant_dtype could actually be not a jnp.dtype (e.g.,
    # jnp.float16), even though it's not supposed to be, we need to
    # wrap with a redundant dtype call to make sure the returned
    # dtype is a instance of `jnp.dtype`
    return jnp.dtype(self.dequant_dtype) if self.dequant_dtype else None

  def _validate_tiling_state(self):
    if self.tiling_state is None:
      return

    if self.is_full():
      assert self.qvalue.shape == tuple(self.tiling_state.tiled_shape), (
          'The shape of the qvalue should be the same as the tiled shape of'
          f' the tiling state. However, {self.qvalue.shape=} and'  # pytype: disable=attribute-error
          f' {self.tiling_state.tiled_shape=}'
      )

  def is_full(self) -> bool:
    return self.qvalue is not None

  def without_qvalue(self) -> Self:
    """Returns a copy of the QTensor without the qvalue."""
    return self.replace(qvalue=None)  # pytype: disable=attribute-error

  def astype(self, dtype: jnp.dtype) -> Self:
    return self.replace(dequant_dtype=dtype)  # pytype: disable=attribute-error

  def quant(self, x) -> Self:
    """Quantizes x into a new QTensor."""
    assert not self.is_full(), 'Already quantized QTensor.'
    assert self.scale is not None, 'Missing scales to be used for quantization.'
    assert isinstance(
        self.scale, list
    ), f'QTensor.scale must be a list of arrays, but got {self.scale}'
    assert isinstance(
        self.bias, list
    ), f'QTensor.bias must be a list of arrays, but got {self.bias}'

    if self.tiling_state is not None:
      x = self.tiling_state.apply(x)

    qvalue = x
    # quant(x) = (x + b) / s
    for b in self.bias:
      qvalue += b

    for s in self.scale:
      # TODO(lew): We could store s_inv for faster activation quantization.
      s_inv = jax.lax.reciprocal(s)
      s_inv = jnp.where(jnp.isinf(s_inv), jnp.ones_like(s_inv), s_inv)
      qvalue *= s_inv

    # TODO(lew): We should apply numerics here, so that 'quant' function
    # Can be considered a part of API.
    return self.replace(qvalue=qvalue)  # pytype: disable=attribute-error

  def dequant(self) -> jnp.ndarray:
    """Dequantizes the QTensor into a jax array."""
    assert self.scale is not None, 'Missing scales when dequantizing a QTensor.'
    assert isinstance(
        self.scale, list
    ), f'QTensor.scale must be a list of arrays, but got {self.scale}'
    assert isinstance(
        self.bias, list
    ), f'QTensor.bias must be a list of arrays, but got {self.bias}'
    msg = (
        'QTensor is manually created without setting a dequant_dtype. It can'
        ' be used in dot_general, but to dequantize you need to set its dtype.'
    )
    assert self.dequant_dtype is not None, msg
    assert self.is_full(), _MSG_NO_QVALUE
    self._validate_tiling_state()

    # pytype: disable=attribute-error
    ret = self.qvalue.astype(self.dequant_dtype)

    # dequant(q) = q * s - b
    for s in self.scale:
      ret *= s

    # Apply bias after all rescaling is done. There may be more biases than
    # scales, e.g. in native asymmetric matmul output dequantization.
    for b in self.bias:
      ret -= b

    if self.tiling_state is not None:
      ret = self.tiling_state.unapply(ret)

    # In case the scale or bias dtypes are not the same as dequant_dtype, and it
    # is a higher precision.
    ret = ret.astype(self.dequant_dtype)
    # pytype: enable=attribute-error
    return ret  # pytype: disable=bad-return-type

  def qvalue_astype(self, dtype) -> Self:
    assert self.is_full(), _MSG_NO_QVALUE
    return self.replace(qvalue=self.qvalue.astype(dtype))  # pytype: disable=attribute-error

  def __getitem__(self, idx: jax_typing.ArrayLike) -> Self:
    """Returns the indexed subtensor on the first axis."""
    assert self.scale_t is None, 'scale_t is not supported in __getitem__'
    assert self.is_full(), _MSG_NO_QVALUE
    self._validate_tiling_state()

    qvalue = self.qvalue[idx]
    scale = [s[idx] for s in self.scale]
    return QTensor(
        qvalue=qvalue,
        scale=scale,
        scale_t=self.scale_t,
        bias=self.bias,
        dequant_dtype=self.dequant_dtype,
    )

  @property
  def ndim(self) -> int:
    assert self.is_full(), _MSG_NO_QVALUE
    self._validate_tiling_state()
    if self.tiling_state is not None:
      return len(self.tiling_state.untiled_shape)
    return self.qvalue.ndim  # pytype: disable=attribute-error

  @property
  def shape(self) -> Sequence[int]:
    assert self.is_full(), _MSG_NO_QVALUE
    self._validate_tiling_state()
    if self.tiling_state is not None:
      return tuple(self.tiling_state.untiled_shape)
    return self.qvalue.shape  # pytype: disable=attribute-error

  def __len__(self) -> int:
    assert self.qvalue is not None, _MSG_NO_QVALUE
    return len(self.qvalue)


def zeros(
    shape: Sequence[int],
    *,
    container_dtype: jnp.dtype,
    dequant_dtype: jnp.dtype = jnp.bfloat16,
) -> QTensor:
  return QTensor(
      qvalue=jnp.zeros(shape, dtype=container_dtype),
      scale=[],
      scale_t=None,
      bias=[],
      dequant_dtype=dequant_dtype,
  )


def zeros_with_scale(
    shape: Sequence[int],
    calibration_axis: Sequence[utils.AxisIdx],
    *,
    container_dtype: jnp.dtype,
    scale_dtype: None | jnp.dtype = None,
    dequant_dtype: jnp.dtype = jnp.bfloat16,
) -> QTensor:
  """Initializes a QTensor with empty qvalue along with empty scale value."""
  scale_shape = list(shape)
  for axis in calibration_axis:
    scale_shape[axis] = 1
  scale_dtype = scale_dtype or dequant_dtype

  # TODO(lew): hardcode dequant_dtype to bf16. This requires updating
  # other libraries to not break their functionality.
  return QTensor(
      qvalue=jnp.zeros(shape, dtype=container_dtype),
      scale=[jnp.ones(scale_shape, dtype=scale_dtype)],
      scale_t=None,
      bias=[],
      dequant_dtype=dequant_dtype,
  )


def partition_spec(
    partitions: Sequence[Any],
    calibration_axis: Sequence[utils.AxisIdx],
    dtype: jnp.dtype,
    *,
    use_bias: bool,
) -> QTensor:
  """Returns a QTensor filled with partition specs."""
  # This function assumes that there is a single scale, and if use_bias=True, a
  # single bias. Both of which are expected to be configured using per-channel
  # quantization.
  scale_partitions = list(partitions)
  for axis in calibration_axis:
    scale_partitions[axis] = None
  if use_bias:
    # Assumes that the bias to be partitioned is the bias from input
    # quantization, (which has singleton dimensions for the calibration_axis),
    # and not the biases used in native output dequantization, of which there
    # may be more than one, and which may have the same shape as the qvalue.
    bias_partition = [jax.sharding.PartitionSpec(*scale_partitions)]
  else:
    # JAX errors upon receiving partition specs for non-existent tensors.
    bias_partition = []
  return QTensor(
      qvalue=jax.sharding.PartitionSpec(*partitions),
      scale=[jax.sharding.PartitionSpec(*scale_partitions)],
      scale_t=None,
      bias=bias_partition,
      dequant_dtype=dtype,
  )


def dynamic_slice(
    operand: QTensor,
    start_indices: Sequence[int],
    slice_sizes: Sequence[int],
) -> QTensor:
  """Dynamically slices the value at start_indices using the given shape."""
  msg = '{attribute} is not supported in the dynamic_slice of a QTensor.'
  assert operand.scale_t is None, msg.format('scale_t')
  assert not operand.bias, msg.format('bias')

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
      qvalue=jax.lax.dynamic_slice(operand.qvalue, start_indices, slice_sizes),
      scale=[get_sliced_scales(s) for s in operand.scale],
      scale_t=None,
      bias=[],
      dequant_dtype=operand.dequant_dtype,
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

  return QTensor(
      qvalue=qvalues,
      scale=scales,
      scale_t=None,
      bias=[],
      dequant_dtype=operand.dequant_dtype,
  )


def update_frame(operand: QTensor, frame: int, update: QTensor) -> QTensor:
  """Updates the value at frame with the given QTensor value."""
  assert operand.ndim == update.ndim + 1
  assert operand.dequant_dtype == update.dequant_dtype, 'Dequant dtype mismatch'

  return QTensor(
      qvalue=operand.qvalue.at[frame].set(update.qvalue),
      scale=[
          target_scale.at[frame].set(update_scale)
          for target_scale, update_scale in zip(operand.scale, update.scale)
      ],
      scale_t=None,
      bias=[
          target_bias.at[frame].set(update_bias)
          for target_bias, update_bias in zip(operand.bias, update.bias)
      ],
      dequant_dtype=operand.dequant_dtype,
  )
