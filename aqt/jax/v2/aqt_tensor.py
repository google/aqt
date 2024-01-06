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
from typing import Any, Callable, Optional
from aqt.jax.v2 import config
from aqt.jax.v2.numerics import no_numerics
import flax.cursor
import flax.struct
import jax
from jax import lax
import jax.numpy as jnp


@flax.struct.dataclass
class QTensor:
  """Quantized tensor."""

  # Quantized (compressed) representation of tensor.
  # Use dequant() method to "decompress" to the original tensor.
  qvalue: jnp.ndarray

  # (scale == None) can be thought as (scale == 1.0)
  # (scale == None) means that qvalue is not quantized and can be used directly.
  scale: Optional[jnp.ndarray]

  # Used in dot_general, transposed scales used in post dot_general scaling.
  # The same comments apply as to scale.
  # We currently keep it here because:
  # - we store scale_t in the checkpoint to avoid transposition per inference.
  # - scale_t is used both in backprop of dot_general and in post-scaling.
  #   We avoid transposing scale twice.
  # Invariant: we never should have a situation where out of scale, scale_t,
  # one is set and one is None.
  # TODO(lew): Move scale_t from QTensor to some dot-general specific type?
  scale_t: Optional[jnp.ndarray]

  def dequant(self) -> jnp.ndarray:
    return self.qvalue if self.scale is None else self.qvalue * self.scale

GradientFn = Callable[..., Any]


def quant(
    x,
    *,
    cfg: config.Tensor,
    calibration_axes,
) -> tuple[QTensor, GradientFn]:
  """The core quantizing function."""
  msg = (
      'use_fake_quant mode is used in tests and it is exactly equal when'
      ' po2_scale == True; Did you forget to set it?'
  )
  assert (not cfg.use_fake_quant) or cfg.po2_scale, msg
  # TODO(lew): We should cast earlier. xhs_q should be in cfg.xhs.dtype
  # TODO(lew): After we implement optimization to not double-quantize,
  #   what would happen if we pass fq value (xhs_q2) in residual?

  if isinstance(cfg.numerics, no_numerics.NoNumerics):
    qt = QTensor(qvalue=x, scale=None, scale_t=None)
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
  numerics_fwd = functools.partial(numerics_fwd, context=cfg.context)

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
  qt = QTensor(qvalue=x_q, scale=scale, scale_t=None)
  return qt, quant_grad


def make_fake_quant(cfg: config.Tensor, calibration_axes=None):
  def fake_quant(x):
    x_q, _ = quant(x, cfg=cfg, calibration_axes=calibration_axes)
    return x_q.dequant()

  return fake_quant
