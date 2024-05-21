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

from typing import Literal, Sequence
from aqt.jax.v2 import aqt_tensor
from aqt.jax.v2 import calibration
from aqt.jax.v2 import utils
from aqt.jax.v2.numerics import int_numerics
from aqt.jax.v2.numerics import no_numerics
from aqt.jax.v2.numerics import numerics
import jax
import jax.numpy as jnp


AbstractAqtNumerics = numerics.AqtNumerics
AbstractAqtCalibration = calibration.Calibration


@utils.flax_slots_kw_only_dataclass
class Quantizer:
  """Configuration of quantization of one tensor."""

  numerics: AbstractAqtNumerics = utils.static_field()
  calib_shared_axes: Sequence[utils.AxisIdx] | Literal["per_tensor"] | None = (
      utils.static_field()
  )
  scale_stop_grad: bool = utils.static_field()
  # noise+clip+round
  # We apply gradient of clip_and_round in bwd pass.
  calibration: type[AbstractAqtCalibration] = utils.static_field()
  # Round up the calibration to power of 2 (po2).
  po2_scale: bool = utils.static_field()
  # TODO(yichizh): Factor out auxilliary dataclasses into a separate file.
  context: utils.Context

  # TODO(yichizh): Need to add type annotation back to cfg.
  def quant(
      self,
      x,
      *,
      calibration_axes,
  ) -> tuple[aqt_tensor.QTensor, aqt_tensor.GradientFn]:
    """The core quantizing function."""
    qt = self.calibrate(x, calibration_axes=calibration_axes)
    return self.calculate_qvalue(x, qt)

  def calibrate(self, x, *, calibration_axes) -> aqt_tensor.QTensor:
    """Create incomplete QTensor with only quantization parameters."""
    if isinstance(self.numerics, no_numerics.NoNumerics):
      return aqt_tensor.QTensor(
          qvalue=x,
          scale=[],
          scale_t=None,
          dequant_dtype=x.dtype,
          numerics=self.numerics,
      )

    dequant_dtype = x.dtype
    # TODO(lew): We should cast earlier. xhs_q should be in cfg.xhs.dtype
    # TODO(lew): After we implement optimization to not double-quantize,
    #   what would happen if we pass fq value (xhs_q2) in residual?
    if self.calib_shared_axes == "per_tensor":
      shared_axes = list(range(x.ndim))
    else:
      shared_axes = self.calib_shared_axes or calibration_axes

    calibrator = self.calibration()
    bound = calibrator.get_bound(x, shared_axes, self.context)
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
        numerics=self.numerics,
    )
    return qt

  def calculate_qvalue(
      self,
      x,
      qt: aqt_tensor.QTensor
  ) -> tuple[aqt_tensor.QTensor, aqt_tensor.GradientFn]:
    """Uses the quantization parameters in qt to quantize x."""
    return qt.quant(x, self.context)


def quantizer_make(
    n_bits: int | None, preserve_max_val: bool = False
) -> Quantizer:
  """Makes Quantizer."""
  if n_bits is None:
    effective_numerics = no_numerics.NoNumerics()
  else:
    pz = False if n_bits == 1 else True
    dtype = utils.infer_dtype_from_bits(n_bits) if pz else None
    effective_numerics = int_numerics.IntNumerics(
        bits=n_bits,
        preserve_zero=pz,
        preserve_max_val=preserve_max_val,
        clip=True,
        round=True,
        noise_fn=None,
        clip_gradient=False,  # This can be disabled when using abs-max scaling.
        dtype=dtype,
    )
  return Quantizer(
      numerics=effective_numerics,
      calib_shared_axes=None,
      scale_stop_grad=True,
      calibration=calibration.AbsMaxCalibration,
      po2_scale=False,
      context=utils.Context(key=None, train_step=None),
  )


def make_fake_quant(quantizer: Quantizer, calibration_axes=None):
  def fake_quant(x):
    x_q, _ = quantizer.quant(x, calibration_axes=calibration_axes)
    return x_q.dequant()

  return fake_quant
