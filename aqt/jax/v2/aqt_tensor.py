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
import itertools
import typing
from typing import Any, Callable, Optional, Self, Sequence, TypeAlias
from aqt.jax.v2 import config
from aqt.jax.v2.numerics import no_numerics
import flax.cursor
import flax.struct
import jax
from jax import lax
import jax.numpy as jnp


GradientFn = Callable[..., Any] | None  # None when there is no numerics


if typing.TYPE_CHECKING:
  # These are needed to avoid static typing complaints for sharding.
  # Concatenating sharding types, e. g. jnp.ndarray | jax.sharding.NamedSharding
  # | jax.sharding.PartitionSpec, will trigger an error to logics where the
  # values are used as jax.array, since they could be sharding.
  ArrayT: TypeAlias = Any
else:
  ArrayT: TypeAlias = jnp.ndarray


@flax.struct.dataclass
class QTensor:
  """Quantized tensor."""

  # Quantized (compressed) representation of tensor.
  # Use dequant() method to "decompress" to the original tensor.
  qvalue: ArrayT

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
  # Default value is added to not to break the existing codebase.
  # TODO(dhchoi): Shall we remove the default dequant_dtype, or leave this?
  dequant_dtype: Optional[jnp.dtype] = flax.struct.field(
      pytree_node=False, default=None
  )

  def dequant(self) -> jnp.ndarray:
    assert self.scale is not None, 'Missing scales when dequantizing a QTensor.'
    msg = (
        'QTensor is manually created without setting a dequant_detype. It can'
        ' be used in dot_general, but to dequantize you need to set its dtype.'
    )
    assert self.dequant_dtype is not None, msg
    ret = self.qvalue
    for scale in self.scale:
      ret = ret.astype(self.dequant_dtype) * scale.astype(self.dequant_dtype)
    return ret

  def qvalue_astype(self, dtype) -> Self:
    return self.replace(qvalue=self.qvalue.astype(dtype))

  def at(self, idx: int):
    return self.__getitem__(idx)

  def __getitem__(self, idx: int):
    """Returns the indexed subtensor on the first axis."""
    assert self.scale_t is None, 'scale_t is not supported in __getitem__'
    # start index is (idx, 0, 0, ...)
    origin_point = (0,) * len(self.qvalue.shape)
    start_idx = (idx,) + tuple(itertools.islice(origin_point, 1, None))
    # slice size is (1, raw_shape, ...)
    slice_sizes = (1,) + tuple(itertools.islice(self.qvalue.shape, 1, None))
    qtensor = QTensor(self.qvalue, self.scale, None, self.dequant_dtype)
    return dynamic_slice(qtensor, start_idx, slice_sizes)

  @property
  def ndim(self) -> int:
    return self.qvalue.ndim

  @property
  def shape(self) -> Sequence[int]:
    return self.qvalue.shape


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


def quant(
    x,
    *,
    cfg: config.Tensor,
    calibration_axes,
    transpose_fn=None,
) -> tuple[QTensor, GradientFn]:
  """The core quantizing function."""
  dequant_dtype = x.dtype
  msg = (
      'Multiply QTensor.qvalue by untransposed QTensor.scale before dot_general'
      '(a.k.a. FakeQuant) is used in tests and it is exactly equal when'
      ' po2_scale == True; Did you forget to set it?'
  )
  assert (
      cfg.dequant_mode != config.DequantMode.THIS_INPUT
  ) or cfg.po2_scale, msg
  # TODO(lew): We should cast earlier. xhs_q should be in cfg.xhs.dtype
  # TODO(lew): After we implement optimization to not double-quantize,
  #   what would happen if we pass fq value (xhs_q2) in residual?

  if isinstance(cfg.numerics, no_numerics.NoNumerics):
    qt = QTensor(qvalue=x, scale=[], scale_t=[], dequant_dtype=dequant_dtype)
    return qt, None
  shared_axes = cfg.calib_shared_axes or calibration_axes
  bound = cfg.calibration.get_bound(x, shared_axes)
  abs_max_mapped_to = cfg.numerics.abs_val_mapped_to()
  scale = abs_max_mapped_to / bound

  if cfg.po2_scale:
    # With floor the biggest value (we are using jnp.max) is in the range of
    # clipping and therefore have a correct gradinet.
    scale = 2 ** jnp.floor(jnp.log2(scale))
  if cfg.scale_stop_grad:
    # TODO(lew): Does not matter in DG, because we are using custom gradient.
    #   We should take that into account somehow.
    scale = lax.stop_gradient(scale)

  x_s = x * scale

  x_q, res = cfg.numerics.vjp_fwd(x_s, cfg.context)
  quant_grad = jax.tree_util.Partial(cfg.numerics.vjp_bwd, res)
  # We are passing quant_grad (and not more) ot the backward pass.
  # That is equivalent to having:
  # scale = stop_gradient(scale)
  #
  # This is not the only possible choice and we intend to allow experimentation.
  # However for today we hardcoded this choice.
  #
  # In order to achevie no-stop-gradiend solution, we should take vjp
  # of a larger piece of code like the whole _scale_quant.
  #
  # TODO(lew): Implement configuration of stop-gradient.
  scale = jax.lax.reciprocal(scale)
  scale_t = None
  if transpose_fn is not None:
    scale_t = [transpose_fn(scale)]

  qt = QTensor(
      qvalue=x_q, scale=[scale], scale_t=scale_t, dequant_dtype=dequant_dtype
  )
  return qt, quant_grad


def make_fake_quant(cfg: config.Tensor, calibration_axes=None):
  def fake_quant(x):
    x_q, _ = quant(x, cfg=cfg, calibration_axes=calibration_axes)
    return x_q.dequant()

  return fake_quant
