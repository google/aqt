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
import functools
from typing import Any, Callable, Optional, Sequence
from aqt.jax.v2 import config
from aqt.jax.v2.numerics import no_numerics
import flax.cursor
import flax.struct
import jax
from jax import lax
import jax.numpy as jnp


GradientFn = Callable[..., Any]


@flax.struct.dataclass
class QTensor:
  """Quantized tensor."""

  # Quantized (compressed) representation of tensor.
  # Use dequant() method to "decompress" to the original tensor.
  qvalue: jnp.ndarray

  # (scale == None) means that scale is unknown/invalid;
  # Otherwise, check dequant(self) for semantics.
  scale: Optional[list[jnp.ndarray]]

  # Used in dot_general, transposed scales used in post dot_general scaling.
  # The same comments apply as to scale.
  # We currently keep it here because:
  # - we store scale_t in the checkpoint to avoid transposition per inference.
  # - scale_t is used both in backprop of dot_general and in post-scaling.
  #   We avoid transposing scale twice.
  # TODO(lew): Move scale_t from QTensor to some dot-general specific type?
  scale_t: Optional[list[jnp.ndarray]]

  def dequant(self, dtype: Optional[jnp.dtype] = None) -> jnp.ndarray:
    assert self.scale is not None
    ret = self.qvalue
    for scale in self.scale:
      if dtype is None:
        ret = ret * scale
      else:
        ret = ret.astype(dtype) * scale.astype(dtype)
    return ret

  @property
  def ndim(self) -> int:
    return self.qvalue.ndim

  @property
  def shape(self) -> Sequence[int]:
    return self.qvalue.shape


def zeros(shape: Sequence[int], qdtype: jnp.dtype) -> QTensor:
  return QTensor(qvalue=jnp.zeros(shape, dtype=qdtype), scale=[], scale_t=[])


def quant(
    x,
    *,
    cfg: config.Tensor,
    calibration_axes,
    transpose_fn=None,
    context: config.Context | None = None
) -> tuple[QTensor, GradientFn]:
  """The core quantizing function."""
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

  if context is None:
    # Try to recover from cfg.context.
    context = cfg.context

  if isinstance(cfg.numerics, no_numerics.NoNumerics):
    qt = QTensor(qvalue=x, scale=[], scale_t=[])
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

  # TODO(lew): custom_vjp should be applied in numerics. We can have
  #   a helper function there to call jax.custom_vjp.
  numerics_fwd = jax.custom_vjp(cfg.numerics.fwd)
  numerics_fwd.defvjp(cfg.numerics.vjp_fwd, cfg.numerics.vjp_bwd)
  numerics_fwd = functools.partial(numerics_fwd, context=context)

  x_q, quant_grad = jax.vjp(numerics_fwd, x_s)
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

  qt = QTensor(qvalue=x_q, scale=[scale], scale_t=scale_t)
  return qt, quant_grad


def make_fake_quant(cfg: config.Tensor, calibration_axes=None):
  def fake_quant(x):
    x_q, _ = quant(
        x, cfg=cfg, calibration_axes=calibration_axes, context=cfg.context
    )
    return x_q.dequant()

  return fake_quant
