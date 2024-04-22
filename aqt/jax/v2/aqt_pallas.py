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

"""AQT for pallas."""

import dataclasses
from typing import Sequence
from aqt.jax.v2 import aqt_tensor
import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp


QTensor = aqt_tensor.QTensor


def _pop_and_append(l, p):
  """Pop l[p] and append at the back."""
  if isinstance(l, list):
    e = l.pop(p)
    l.append(e)
  elif isinstance(l, tuple):
    e = l[p]
    l = (*l[:p], e, *l[p+1:])
  return l


def quant_blockwisely(
    x: jax.Array,
    n_bits: int,
    calibration_axes: Sequence[int],
    block_spec: pl.BlockSpec,
) -> tuple[QTensor, QTensor]:
  """Quantize x block-wisely according to block_spec.

  x is quantized block-wisely (a.k.a subchannel) on the calibration axes, and
  the size of block of each axis is determined by block_spec.block_shape[axis]

  Args:
    x: input tensor
    n_bits: the precision for quantization.
    calibration_axes: the calibration axes.
    block_spec: Pallas BlockSpec of the input x

  Returns:
    A tuple of QTensor and block spec of that QTensor.
  """

  if n_bits not in [4, 8]:
    raise ValueError('n_bits must be either 4 or 8')

  # TODO(wppark): use aqt_quantizer.Quantizer instead of code written from
  # scratch.
  tiled_x_shape = []
  for axis, ndim in enumerate(x.shape):
    if axis in calibration_axes:
      tiled_x_shape += [
          ndim // block_spec.block_shape[axis],
          block_spec.block_shape[axis],
      ]
    else:
      tiled_x_shape += [ndim]

  tiled_x = jnp.reshape(x, tiled_x_shape)
  tiled_calibration_axes = [
      (i + 1) + idx for i, idx in enumerate(calibration_axes)
  ]

  abs_max = jnp.max(
      jnp.abs(tiled_x), axis=tiled_calibration_axes, keepdims=True
  )
  tiled_scale = abs_max / (2 ** (n_bits - 1) - 1)

  tiled_qx = jax.lax.round(
      tiled_x / tiled_scale, jax.lax.RoundingMethod.TO_NEAREST_EVEN
  )
  tiled_qx = tiled_qx.astype(jnp.int8 if n_bits == 8 else jnp.int4)
  tiled_qx = jnp.reshape(tiled_qx, x.shape)

  qvalue = jnp.reshape(tiled_qx, x.shape)
  scale = jnp.squeeze(tiled_scale, axis=tiled_calibration_axes)

  scale_block_shape = tuple([
      1 if axis in calibration_axes else ndim
      for axis, ndim in enumerate(block_spec.block_shape)
  ])

  # transpose scale such that:
  # - the size of last dimension should be bigger 128.
  # - the size second last dimension is 1.

  # find the inner most dimension that its size is multiples of 128.
  large_dim = 0
  for axis, ndim in enumerate(scale.shape):
    if ndim >= 128 and ndim % 128 == 0:
      large_dim = axis

  scale_permute_axis = list(range(scale.ndim))
  # make large dim as the last dimension
  scale_permute_axis = _pop_and_append(scale_permute_axis, large_dim)

  # transpose scale and its block shape accordingly
  scale = jnp.transpose(scale, scale_permute_axis)
  scale_block_shape = [scale_block_shape[ax] for ax in scale_permute_axis]

  # make the size of second last dimension to be 1
  is_expand_dims = scale.shape[-2] != 1
  if is_expand_dims:
    scale = jnp.expand_dims(scale, axis=-2)
    scale_permute_axis.insert(len(scale_permute_axis) - 1, -1)
    scale_block_shape = (*scale_block_shape[:-1], 1, scale_block_shape[-1])

  def scale_index_map(*args):
    index = block_spec.index_map(*args)
    index = _pop_and_append(index, large_dim)
    if is_expand_dims:
      index = (*index[:-1], 0, index[-1])
    return index

  scale_block_spec = pl.BlockSpec(
      index_map=scale_index_map,
      block_shape=scale_block_shape,
  )
  qx = QTensor(
      qvalue=qvalue,
      scale=[scale],
      scale_t=None,
      dequant_dtype=scale.dtype,
      scale_permute_axis=[scale_permute_axis],
  )

  qx_block_spec = dataclasses.replace(
      qx,
      qvalue=block_spec,
      scale=[scale_block_spec],
  )
  return qx, qx_block_spec


def materialize_qtensor(qtensor: QTensor) -> QTensor:
  """Materialize QTensor of MemoryRef of pallas into QTensor of jax.Array."""
  qvalue = qtensor.qvalue
  scale = qtensor.scale
  scale_t = qtensor.scale_t

  if qvalue is not None:
    qvalue = qvalue[...]
  if scale is not None:
    scale = [s[...] for s in scale]
  if scale_t is not None:
    scale_t = [st[...] for st in scale_t]

  return qtensor.replace(qvalue=qvalue, scale=scale, scale_t=scale_t)
