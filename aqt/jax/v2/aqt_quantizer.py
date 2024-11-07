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
from aqt.jax.v2 import tiled_dot_general
from aqt.jax.v2 import utils
from aqt.jax.v2.numerics import fp8_numerics
from aqt.jax.v2.numerics import no_numerics
from aqt.jax.v2.numerics import numerics
from aqt.jax.v2.numerics import utils as numerics_utils
import jax
import jax.numpy as jnp


AbstractAqtNumerics = numerics.AqtNumerics
AbstractAqtCalibration = calibration.Calibration

AxisTiling = tiled_dot_general.AxisTiling
TilingState = tiled_dot_general.TilingState


@utils.flax_slots_kw_only_dataclass
class Quantizer:
  """Configuration of quantization of one tensor."""

  numerics: AbstractAqtNumerics = utils.static_field()
  calib_shared_axes: None | Sequence[utils.AxisIdx] | Literal["per_tensor"] = (
      utils.static_field()
  )
  scale_stop_grad: bool = utils.static_field()
  scale_dtype: None | jnp.dtype = utils.static_field(default=None)
  # noise+clip+round
  # We apply gradient of clip_and_round in bwd pass.
  calibration: None | type[AbstractAqtCalibration] = utils.static_field(
      default=None
  )
  _calibrator: None | AbstractAqtCalibration = utils.static_field(default=None)
  # TODO(yichizh): Factor out auxiliary dataclasses into a separate file.
  context: utils.Context

  # we need to speed up this initialization for the backward pass to happen
  # outside of bwd pass.
  def init_calibration(self):
    assert self._calibrator is None, "second call to self.init_calibration()"
    if self.calibration is not None:
      self._calibrator = self.calibration(dtype=self.scale_dtype)
      self._calibrator.init_calibration()

  # TODO(yichizh): Need to add type annotation back to cfg.
  def quant(
      self,
      x,
      *,
      calibration_axes: None | Sequence[utils.AxisIdx],
      tiling_state: None | TilingState = None,
  ) -> tuple[aqt_tensor.QTensor, aqt_tensor.GradientFn]:
    """The core quantizing function."""
    qt = self.calibrate(
        x, calibration_axes=calibration_axes, tiling_state=tiling_state
    )
    qt, quant_grad = self.calculate_qvalue(x, qt)
    return qt, quant_grad

  def calibrate(
      self,
      x,
      *,
      calibration_axes: None | Sequence[utils.AxisIdx],
      tiling_state: None | TilingState = None,
  ) -> aqt_tensor.QTensor:
    """Creates incomplete QTensor with only quantization parameters.

    The tiling state is used to tile the input tensor and change the calibration
    axes accordingly. When axis is tiled, it is split into multiple tiles. Each
    tile shares the same quantization parameters like scale factor. On the other
    hand, if the axis is not tiled, the whole axis shares the same quantization
    parameters. This tiling will increase the granularity of calibration
    reducing the numeric error from quantization.

    Args:
      x: The input tensor to be calibrated.
      calibration_axes: The axes to calibrate.
      tiling_state: The tiling state of the input tensor.

    Returns:
      An incomplete QTensor with only quantization parameters.
    """

    if tiling_state:
      # Tile `x` and change calibration axes according to the tiling.
      x = tiling_state.apply(x)
      _, calibration_axes = tiling_state.to_tiled_axes_transposed(
          calibration_axes
      )

    if isinstance(self.numerics, no_numerics.NoNumerics):
      qt = aqt_tensor.QTensor(
          qvalue=x,
          scale=[],
          scale_t=None,
          bias=[],
          dequant_dtype=x.dtype,
          tiling_state=tiling_state,
      )
      return qt

    dequant_dtype = x.dtype
    # TODO(lew): We should cast earlier. xhs_q should be in cfg.xhs.dtype
    # TODO(lew): After we implement optimization to not double-quantize,
    #   what would happen if we pass fq value (xhs_q2) in residual?
    if self.calib_shared_axes == "per_tensor":
      shared_axes = list(range(x.ndim))
    else:
      shared_axes = self.calib_shared_axes or calibration_axes

    msg = (
        f"{self.calibration = } must be set if {self.numerics = } is not"
        " NoNumerics"
    )
    assert self.calibration is not None, msg
    assert self._calibrator is not None, "forgot self.init_calibration()?"

    scale, bias, sparsity_mask = (
        self._calibrator.get_scale_and_bias_and_sparsity(
            x, shared_axes, self.numerics, self.context
        )
    )
    if self.scale_stop_grad:
      # TODO(lew): Does not matter in DG, because we are using custom gradient.
      #   We should take that into account somehow.
      scale = jax.lax.stop_gradient(scale)

    qt = aqt_tensor.QTensor(
        qvalue=None,
        sparsity_mask=sparsity_mask,
        scale=scale,
        scale_t=None,
        bias=bias,
        dequant_dtype=dequant_dtype,
        tiling_state=tiling_state,
    )
    return qt

  def calculate_qvalue(
      self, x, qt: aqt_tensor.QTensor
  ) -> tuple[aqt_tensor.QTensor, aqt_tensor.GradientFn]:
    """Uses the quantization parameters in qt to quantize x."""
    if isinstance(self.numerics, no_numerics.NoNumerics):
      return qt, None

    # TODO: b/333984742 - make numeric as a member of QTensor, and put
    # numerics-related logics into the QTensor.
    qt = qt.quant(x)

    # TODO(lew): A logical thing would be if this call was part of
    # QTensor.quant.
    x_q, res = self.numerics.vjp_fwd(qt.qvalue, self.context)
    quant_grad = jax.tree_util.Partial(self.numerics.vjp_bwd, res)

    qt.qvalue = x_q
    return qt, quant_grad


def quantizer_make(
    n_bits: None | int | fp8_numerics.FP8Dtype,
    preserve_max_val: bool = False,
    initialize_calibration: bool = True,
    scale_stop_grad: bool = True,
    scale_dtype: None | jnp.dtype = None,
) -> Quantizer:
  """Makes Quantizer."""
  effective_numerics = numerics_utils.get_numerics(n_bits, preserve_max_val)

  if n_bits is None:
    calibration_cls = None
  else:
    calibration_cls = calibration.AbsMaxCalibration

  quantizer = Quantizer(
      numerics=effective_numerics,
      calib_shared_axes=None,
      scale_stop_grad=scale_stop_grad,
      scale_dtype=scale_dtype,
      calibration=calibration_cls,
      context=utils.Context(key=None, train_step=None),
  )
  # TODO(lew): We should try to move to to class constructor or post-init.
  # We currently need to call because bwd pass is too late for initialization.
  if initialize_calibration:
    quantizer.init_calibration()
  return quantizer


def make_fake_quant(quantizer: Quantizer, calibration_axes=None):
  def fake_quant(x):
    x_q, _ = quantizer.quant(x, calibration_axes=calibration_axes)
    return x_q.dequant()

  return fake_quant
