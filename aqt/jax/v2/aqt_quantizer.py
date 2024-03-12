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
"""Configuration dataclasses."""

from typing import Literal
from aqt.jax.v2 import aqt_tensor
from aqt.jax.v2 import calibration
from aqt.jax.v2 import utils
from aqt.jax.v2.numerics import no_numerics
from aqt.jax.v2.numerics import numerics
import jax
import jax.numpy as jnp


AbstractAqtNumerics = numerics.AqtNumerics
AbstractAqtCalibration = calibration.Calibration


@utils.flax_slots_dataclass
class Context:
  key: jax.Array | None = utils.dynamic_field()
  train_step: int | None = utils.dynamic_field()


@utils.flax_slots_dataclass
class Quantizer:
  """Configuration of quantization of one tensor."""

  numerics: AbstractAqtNumerics = utils.static_field()
  calib_shared_axes: list[int] | Literal["per_tensor"] | None = (
      utils.static_field()
  )
  scale_stop_grad: bool = utils.static_field()
  # noise+clip+round
  # We apply gradient of clip_and_round in bwd pass.
  calibration: AbstractAqtCalibration = utils.static_field()
  # Round up the calibration to power of 2 (po2).
  po2_scale: bool = utils.static_field()
  # TODO(yichizh): Factor out auxilliary dataclasses into a separate file.
  context: Context

  # TODO(yichizh): Need to add type annotation back to cfg.
  def quant(
      self,
      x,
      *,
      calibration_axes,
  ) -> tuple[aqt_tensor.QTensor, aqt_tensor.GradientFn]:
    """The core quantizing function."""
    if isinstance(self.numerics, no_numerics.NoNumerics):
      qt = aqt_tensor.QTensor(
          qvalue=x, scale=[], scale_t=[], dequant_dtype=x.dtype
      )
      return qt, None

    qt = self.calculate_scale(x, calibration_axes=calibration_axes)
    qt, quant_grad = self.calculate_qvalue(x, qt)
    return qt, quant_grad

  def calculate_scale(
      self,
      x,
      *,
      calibration_axes,
  ) -> aqt_tensor.QTensor:
    """Create incomplete QTensor with only quantization parameters."""
    dequant_dtype = x.dtype
    # TODO(lew): We should cast earlier. xhs_q should be in cfg.xhs.dtype
    # TODO(lew): After we implement optimization to not double-quantize,
    #   what would happen if we pass fq value (xhs_q2) in residual?
    if self.calib_shared_axes == "per_tensor":
      shared_axes = list(range(x.ndim))
    else:
      shared_axes = self.calib_shared_axes or calibration_axes
    bound = self.calibration.get_bound(x, shared_axes)
    abs_max_mapped_to = self.numerics.abs_val_mapped_to()
    scale = bound / abs_max_mapped_to

    if self.po2_scale:
      # With floor the biggest value (we are using jnp.max) is in the range of
      # clipping and therefore have a correct gradinet.
      scale = 2 ** jnp.floor(jnp.log2(jax.lax.reciprocal(scale)))
      scale = jax.lax.reciprocal(scale)
    if self.scale_stop_grad:
      # TODO(lew): Does not matter in DG, because we are using custom gradient.
      #   We should take that into account somehow.
      scale = jax.lax.stop_gradient(scale)

    qt = aqt_tensor.QTensor(
        qvalue=None,
        scale=[scale],
        scale_t=None,
        dequant_dtype=dequant_dtype,
    )
    return qt

  def calculate_qvalue(
      self,
      x,
      qt: aqt_tensor.QTensor
  ) -> tuple[aqt_tensor.QTensor, aqt_tensor.GradientFn]:
    """Uses the quantization parameters in qt to quantize x."""
    qt = qt.quant(x)

    # TODO(lew): A logical thing would be if this call was part of
    # QTensor.quant.
    x_q, res = self.numerics.vjp_fwd(qt.qvalue, self.context)
    quant_grad = jax.tree_util.Partial(self.numerics.vjp_bwd, res)

    qt = qt.replace(qvalue=x_q)
    return qt, quant_grad


def make_fake_quant(cfg, calibration_axes=None):
  def fake_quant(x):
    x_q, _ = cfg.quantizer.quant(x, calibration_axes=calibration_axes)
    return x_q.dequant()

  return fake_quant
