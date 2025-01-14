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

# pylint: disable=g-importing-member
# pylint: disable=unused-import
# pylint: disable=g-explicit-bool-comparison

import copy
import functools
from typing import Literal, Sequence, TypeAlias

from aqt.jax.v2 import aqt_dot_general
from aqt.jax.v2 import aqt_quantizer
from aqt.jax.v2 import calibration
from aqt.jax.v2 import stochastic_rounding
from aqt.jax.v2 import utils
# Temporary re-export from aqt.jax.v2.aqt_dot_general
# TODO(lew): Remove these imports, use setters instead
from aqt.jax.v2.aqt_conv_general import conv_general_dilated_make
from aqt.jax.v2.aqt_dot_general import CalibrationMode
from aqt.jax.v2.aqt_dot_general import DequantMode
from aqt.jax.v2.aqt_dot_general import dot_general_make
from aqt.jax.v2.aqt_dot_general import dot_general_raw_make
from aqt.jax.v2.aqt_dot_general import DotGeneral
from aqt.jax.v2.aqt_dot_general import DotGeneralRaw
from aqt.jax.v2.aqt_dot_general import dtypes_allowed_for_int32_accum
from aqt.jax.v2.aqt_dot_general import LocalAqt
from aqt.jax.v2.aqt_dot_general import Tensor
from aqt.jax.v2.aqt_quantizer import quantizer_make
from aqt.jax.v2.numerics import fp8_numerics
from aqt.jax.v2.numerics import fp_numerics
from aqt.jax.v2.numerics import int_numerics
from aqt.jax.v2.numerics import no_numerics
from aqt.jax.v2.numerics import numerics
from aqt.jax.v2.numerics import utils as numerics_utils
import jax
import jax.numpy as jnp


################################################################################
# Functions below are auxiliary config attribute setters.

# SKIP can be used as an argument to some set_xyz functions.
# It signals that set_xyz should make no changes to that option.
SKIP = 'skip'
SkipT: TypeAlias = Literal[SKIP]


def _split_key(key: None | jax.Array, num_splits: int):
  default = (None,) * num_splits
  return default if key is None else jax.random.split(key, num_splits)


def set_context(
    cfg: DotGeneral,
    key: None | jax.Array,
    train_step: None | int,
    lhs_quant_mode: utils.QuantMode = utils.QuantMode.TRAIN,
    rhs_quant_mode: utils.QuantMode = utils.QuantMode.TRAIN,
):
  """Set context with prng keys and train_steps for dot_general config."""

  def set_dg_raw_context(cfg_raw: DotGeneralRaw, key: None | jax.Array):
    key1, key2 = _split_key(key, num_splits=2)
    lhs_context = utils.Context(
        key=key1, train_step=train_step, quant_mode=lhs_quant_mode
    )
    rhs_context = utils.Context(
        key=key2, train_step=train_step, quant_mode=rhs_quant_mode
    )
    cfg_raw.dg_quantizer.set_context(lhs_context, rhs_context)

  key_fwd, key_dlhs, key_drhs = _split_key(key, num_splits=3)
  ret_cfg = copy.deepcopy(cfg)
  set_dg_raw_context(ret_cfg.fwd, key_fwd)
  set_dg_raw_context(ret_cfg.dlhs, key_dlhs)
  set_dg_raw_context(ret_cfg.drhs, key_drhs)
  return ret_cfg


def set_fwd_dequant_mode(
    cfg: DotGeneral,
    *,
    lhs_dequant_mode: None | DequantMode = None,
    rhs_dequant_mode: None | DequantMode = None,
):
  if lhs_dequant_mode is not None:
    cfg.fwd.lhs.dequant_mode = lhs_dequant_mode
  if rhs_dequant_mode is not None:
    cfg.fwd.rhs.dequant_mode = rhs_dequant_mode


def set_fwd_calibration_mode(
    cfg: DotGeneral,
    *,
    lhs_calibration_mode: CalibrationMode | SkipT = SKIP,
    rhs_calibration_mode: CalibrationMode | SkipT = SKIP,
):
  if lhs_calibration_mode != SKIP:
    cfg.fwd.lhs.calibration_mode = lhs_calibration_mode
  if rhs_calibration_mode != SKIP:
    cfg.fwd.rhs.calibration_mode = rhs_calibration_mode


def set_numerics(
    cfg: DotGeneralRaw,
    lhs_numerics: numerics.AqtNumerics,
    rhs_numerics: numerics.AqtNumerics,
):
  """Set numerics for DotGeneralRaw config."""
  assert isinstance(
      cfg.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )
  cfg.dg_quantizer.lhs.numerics = lhs_numerics
  cfg.dg_quantizer.rhs.numerics = rhs_numerics
  if (
      lhs_numerics.get_dtype() in dtypes_allowed_for_int32_accum
      and rhs_numerics.get_dtype() in dtypes_allowed_for_int32_accum
  ):
    cfg.dg_accumulator_dtype = jnp.int32
  elif (
      lhs_numerics.get_dtype() in fp8_numerics.fp8_map.values()
      or rhs_numerics.get_dtype() in fp8_numerics.fp8_map.values()
  ):
    cfg.dg_accumulator_dtype = jnp.float32
  else:
    cfg.dg_accumulator_dtype = None


def set_fwd_rhs_dtype_int2(cfg: DotGeneral):
  """A special setter for int2 weights."""
  # Since XLA only supports int2 casting with an input shape of a multiple
  # of 128, we use this setter to enable int2 dtype.
  # Remove this setter and enable int2 in utils.infer_dtype_from_bits()
  # when XLA supports general int2 casting.
  assert isinstance(
      cfg.fwd.dg_quantizer.rhs.numerics, int_numerics.IntSymmetric
  )
  assert cfg.fwd.dg_quantizer.rhs.numerics.bits == 2
  # Disable pytype check since jnp.int2 is only dynamically to jax
  # when ml_dtypes package has it.
  cfg.fwd.dg_quantizer.rhs.numerics.dtype = jnp.int2  # pytype: disable=module-attr


def set_accumulator_dtype(
    cfg: DotGeneral,
    fwd_dtype: None | jnp.dtype | SkipT,
    dlhs_dtype: None | jnp.dtype | SkipT,
    drhs_dtype: None | jnp.dtype | SkipT,
):
  if fwd_dtype != SKIP:
    cfg.fwd.dg_accumulator_dtype = fwd_dtype
  if dlhs_dtype != SKIP:
    cfg.dlhs.dg_accumulator_dtype = dlhs_dtype
  if drhs_dtype != SKIP:
    cfg.drhs.dg_accumulator_dtype = drhs_dtype


def set_stochastic_rounding(
    cfg: DotGeneral,
    # Typically we have (but it's a caller's responsibility to check):
    # - vjp_lhs_stochastic_rounding is referring to the gradient and
    # - vjp_rhs_stochastic_rounding is referring to the activations/weights.
    vjp_lhs_stochastic_rounding: bool,
    vjp_rhs_stochastic_rounding: bool,
    implementation: str,
    noise_sharing_axes: Sequence[int] = (),
):
  """Configure stochastic rounding implementation."""
  noise_implementations = {
      'jax.uniform': stochastic_rounding.JaxUniform(),
      'custom-1': stochastic_rounding.RandomCenteredUniform(),
  }
  msg = f'{implementation} not supported.'
  assert implementation in noise_implementations.keys(), msg
  if noise_sharing_axes:
    noise_fn = functools.partial(
        noise_implementations[implementation],
        noise_sharing_axes=noise_sharing_axes,
    )
  else:
    # for backward compatibility of the config tests.
    noise_fn = noise_implementations[implementation]
  assert isinstance(
      cfg.dlhs.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )
  assert isinstance(
      cfg.drhs.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )

  def _set_numerics_noise_fn(
      dg_quantizer: aqt_dot_general.DefaultDotGeneralQuantizer,
      is_lhs_sr: bool,
      is_rhs_sr: bool,
  ) -> None:
    # set stochastic noise for each dg_quantizer
    def _set_noise_fn(
        quantizer: aqt_quantizer.Quantizer,
        is_sr: bool,
    ) -> None:
      # set stochastic noise for each side of a dg_quantizer
      if isinstance(quantizer.numerics, fp8_numerics.Fp8Numerics):
        quantizer.numerics.stochastic_rounding = is_sr
      else:
        quantizer.numerics.noise_fn = noise_fn if is_sr else None

    _set_noise_fn(dg_quantizer.lhs, is_lhs_sr)
    _set_noise_fn(dg_quantizer.rhs, is_rhs_sr)

  is_lhs_sr = vjp_lhs_stochastic_rounding
  is_rhs_sr = vjp_rhs_stochastic_rounding
  _set_numerics_noise_fn(cfg.dlhs.dg_quantizer, is_lhs_sr, is_rhs_sr)
  _set_numerics_noise_fn(cfg.drhs.dg_quantizer, is_lhs_sr, is_rhs_sr)


def set_constant_calibration(
    cfg: DotGeneral, bound: float = 1.0, bias: None | float = None
):
  """Sets the static bound for calibration."""
  calibration_cls = functools.partial(
      calibration.ConstantCalibration, bound=bound, bias=bias
  )

  assert isinstance(
      cfg.fwd.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )
  assert isinstance(
      cfg.dlhs.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )
  assert isinstance(
      cfg.drhs.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )

  cfg.fwd.dg_quantizer.lhs.calibration = calibration_cls
  cfg.fwd.dg_quantizer.rhs.calibration = calibration_cls
  cfg.dlhs.dg_quantizer.lhs.calibration = calibration_cls
  cfg.dlhs.dg_quantizer.rhs.calibration = calibration_cls
  cfg.drhs.dg_quantizer.lhs.calibration = calibration_cls
  cfg.drhs.dg_quantizer.rhs.calibration = calibration_cls


def set_local_aqt(
    cfg: DotGeneral,
    fwd_local_aqt: None | SkipT | LocalAqt,
    dlhs_local_aqt: None | SkipT | LocalAqt,
    drhs_local_aqt: None | SkipT | LocalAqt,
):
  if fwd_local_aqt != SKIP:
    cfg.fwd.local_aqt = fwd_local_aqt
  if dlhs_local_aqt != SKIP:
    cfg.dlhs.local_aqt = dlhs_local_aqt
  if drhs_local_aqt != SKIP:
    cfg.drhs.local_aqt = drhs_local_aqt


def set_use_fwd_quant(
    cfg: DotGeneral,
    dlhs_use_fwd_quant: None | bool | SkipT,
    drhs_use_fwd_quant: None | bool | SkipT,
):
  """Enable resusing of fwd pass quantization for backprop."""
  msg = 'use_fwd_quant is incompatible with use_mid_quant right now.'
  assert cfg.fwd.dg_quantizer.lhs_mid_alpha is None, msg
  assert cfg.fwd.dg_quantizer.rhs_mid_alpha is None, msg
  assert cfg.dlhs.dg_quantizer.lhs_mid_alpha is None, msg
  assert cfg.dlhs.dg_quantizer.rhs_mid_alpha is None, msg
  assert cfg.drhs.dg_quantizer.lhs_mid_alpha is None, msg
  assert cfg.drhs.dg_quantizer.rhs_mid_alpha is None, msg
  if dlhs_use_fwd_quant != SKIP:
    cfg.dlhs.rhs.use_fwd_quant = dlhs_use_fwd_quant
  if drhs_use_fwd_quant != SKIP:
    cfg.drhs.rhs.use_fwd_quant = drhs_use_fwd_quant


def set_use_mid_quant(
    cfg: DotGeneral,
    fwd_mid_alpha_both: SkipT | float,
    dlhs_mid_alpha_both: SkipT | float,
    drhs_mid_alpha_both: SkipT | float,
):
  """Enable middle quantization. Variant of SmoothQuant / AWQ."""
  assert isinstance(
      cfg.fwd.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )
  assert isinstance(
      cfg.dlhs.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )
  assert isinstance(
      cfg.drhs.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )

  msg = 'use_fwd_quant is incompatible with use_mid_quant right now.'
  assert not cfg.dlhs.rhs.use_fwd_quant, msg
  assert not cfg.drhs.rhs.use_fwd_quant, msg

  @utils.flax_slots_kw_only_dataclass
  class DummyNumerics(numerics.AqtNumerics):
    """DummyNumerics for mid-quantization."""

    def get_quant_bound(self):
      return 1.0

    def get_dtype(self):
      assert False, 'Should not request dtype for mid-quantization.'

    def vjp_fwd(self, x, context):
      res = ()
      return x, res

    def vjp_bwd(self, res, grad):
      assert res == ()
      return grad

  calibration_cls = calibration.AbsMaxCalibration
  if fwd_mid_alpha_both != SKIP:
    cfg.fwd.dg_quantizer.lhs_mid.numerics = DummyNumerics()
    cfg.fwd.dg_quantizer.rhs_mid.numerics = DummyNumerics()
    cfg.fwd.dg_quantizer.lhs_mid.calibration = calibration_cls
    cfg.fwd.dg_quantizer.rhs_mid.calibration = calibration_cls
    cfg.fwd.dg_quantizer.lhs_mid_alpha = fwd_mid_alpha_both
    cfg.fwd.dg_quantizer.rhs_mid_alpha = fwd_mid_alpha_both
  if dlhs_mid_alpha_both != SKIP:
    cfg.dlhs.dg_quantizer.lhs_mid.numerics = DummyNumerics()
    cfg.dlhs.dg_quantizer.rhs_mid.numerics = DummyNumerics()
    cfg.dlhs.dg_quantizer.lhs_mid.calibration = calibration_cls
    cfg.dlhs.dg_quantizer.rhs_mid.calibration = calibration_cls
    cfg.dlhs.dg_quantizer.lhs_mid_alpha = dlhs_mid_alpha_both
    cfg.dlhs.dg_quantizer.rhs_mid_alpha = dlhs_mid_alpha_both
  if drhs_mid_alpha_both != SKIP:
    cfg.drhs.dg_quantizer.lhs_mid.numerics = DummyNumerics()
    cfg.drhs.dg_quantizer.rhs_mid.numerics = DummyNumerics()
    cfg.drhs.dg_quantizer.lhs_mid.calibration = calibration_cls
    cfg.drhs.dg_quantizer.rhs_mid.calibration = calibration_cls
    cfg.drhs.dg_quantizer.lhs_mid_alpha = drhs_mid_alpha_both
    cfg.drhs.dg_quantizer.rhs_mid_alpha = drhs_mid_alpha_both


def set_int_numerics_preserve_zero(cfg: DotGeneral, preserve_zero: bool):
  """Set preserve_zero for int_numerics."""
  assert isinstance(
      cfg.fwd.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )
  assert isinstance(
      cfg.dlhs.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )
  assert isinstance(
      cfg.drhs.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )

  for dot_general_raw in [cfg.fwd, cfg.dlhs, cfg.drhs]:
    dg_quantizer = dot_general_raw.dg_quantizer
    for q_numerics in [dg_quantizer.lhs.numerics, dg_quantizer.rhs.numerics]:
      if isinstance(q_numerics, int_numerics.IntSymmetric):
        q_numerics.preserve_zero = preserve_zero
        updated_dtype = (
            utils.infer_dtype_from_bits(q_numerics.bits)
            if preserve_zero
            else None
        )
        q_numerics.dtype = updated_dtype


def set_auto_calib_scale(
    cfg: DotGeneral, auto_clip_search_config: utils.AutoScaleSearchConfig
) -> None:
  """Update `cfg`'s quantizers' calibration to use auto clipping search.

  Currently only supports the weights (rhs) of `DotGeneral`, since the iterative
  process of finding the scale tensors might be too slow for the activations
  (lhs).

  Args:
    cfg: The config to be updated.
    auto_clip_search_config: The config for auto clipping search.
  """
  assert isinstance(
      cfg.fwd.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )
  assert isinstance(
      cfg.dlhs.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )
  assert isinstance(
      cfg.drhs.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )

  for dot_general_raw in [cfg.fwd, cfg.dlhs, cfg.drhs]:
    quantizer = dot_general_raw.dg_quantizer.rhs
    # TODO(lew): Remove partial inspection wherever possible.
    # Partial inspection is needed because the current implementation of delayed
    # calibration initialization requires the configuration to be set via
    # functools.partial.
    keywords = {}
    if isinstance(quantizer.calibration, functools.partial):
      keywords = quantizer.calibration.keywords
    keywords.update(auto_clip_search_config=auto_clip_search_config)
    quantizer.calibration = functools.partial(
        calibration.SnrBasedAutoCalibration, **keywords
    )


def set_absmax_calib_scale(cfg: DotGeneral, scale: float):
  """Set clipping_scale and clip_gradient for AbsMaxCalibration quantizers.

  Does not modify the configuration for quantizers with other calibration
  classes or None calibration.

  Args:
    cfg: The config to be updated.
    scale: The clipping scale.
  """
  assert isinstance(
      cfg.fwd.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )
  assert isinstance(
      cfg.dlhs.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )
  assert isinstance(
      cfg.drhs.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )

  for dot_general_raw in [cfg.fwd, cfg.dlhs, cfg.drhs]:
    dg_quantizer = dot_general_raw.dg_quantizer
    for quantizer in [dg_quantizer.lhs, dg_quantizer.rhs]:
      calibration_cls = quantizer.calibration
      if calibration_cls is None:
        continue

      # TODO(lew): Remove partial inspection wherever possible.
      # Partial inspection is needed because the current implementation of
      # delayed calibration initialization requires the configuration to be set
      # via functools.partial.
      keywords = {}
      if isinstance(calibration_cls, functools.partial):
        keywords = calibration_cls.keywords
        calibration_cls = calibration_cls.func
      keywords.update(clipping_scale=scale)
      assert calibration_cls == calibration.AbsMaxCalibration, (
          'clipping_scale is only available in AbsMaxCalibration, while'
          f' {quantizer.calibration} is used in current config.'
      )
      quantizer.calibration = functools.partial(
          calibration.AbsMaxCalibration,
          **keywords,
      )
      if scale < 1.0 and isinstance(
          quantizer.numerics, int_numerics.IntSymmetric
      ):
        quantizer.numerics.clip_gradient = True


def set_bits(
    cfg: DotGeneral,
    fwd_lhs_bit: None | int | fp8_numerics.FP8Dtype,
    fwd_rhs_bit: None | int | fp8_numerics.FP8Dtype,
    dlhs_lhs_bit: None | int | fp8_numerics.FP8Dtype,
    dlhs_rhs_bit: None | int | fp8_numerics.FP8Dtype,
    drhs_lhs_bit: None | int | fp8_numerics.FP8Dtype,
    drhs_rhs_bit: None | int | fp8_numerics.FP8Dtype,
) -> DotGeneral:
  """Set quant bits for dot_general. Overwrites with AbsMaxCalibration."""
  calibration_cls = calibration.AbsMaxCalibration

  set_numerics(
      cfg.fwd,
      numerics_utils.get_numerics(fwd_lhs_bit),
      numerics_utils.get_numerics(fwd_rhs_bit),
  )
  if fwd_lhs_bit is not None:
    cfg.fwd.dg_quantizer.lhs.calibration = calibration_cls
  if fwd_rhs_bit is not None:
    cfg.fwd.dg_quantizer.rhs.calibration = calibration_cls

  set_numerics(
      cfg.dlhs,
      numerics_utils.get_numerics(dlhs_lhs_bit),
      numerics_utils.get_numerics(dlhs_rhs_bit),
  )
  if dlhs_lhs_bit is not None:
    cfg.dlhs.dg_quantizer.lhs.calibration = calibration_cls
  if dlhs_rhs_bit is not None:
    cfg.dlhs.dg_quantizer.rhs.calibration = calibration_cls

  set_numerics(
      cfg.drhs,
      numerics_utils.get_numerics(drhs_lhs_bit),
      numerics_utils.get_numerics(drhs_rhs_bit),
  )
  if drhs_lhs_bit is not None:
    cfg.drhs.dg_quantizer.lhs.calibration = calibration_cls
  if drhs_rhs_bit is not None:
    cfg.drhs.dg_quantizer.rhs.calibration = calibration_cls

  # use_fwd_quant is by default set to False if fwd pass is quantized.
  # This is to make the configuration logically correct,
  # i.e., use_fwd_quant cannot be None when fwd is quantized.
  # It is user's responsibility to further choose between False and True.
  dlhs_use_fwd_quant = False if fwd_rhs_bit is not None else SKIP
  drhs_use_fwd_quant = False if fwd_lhs_bit is not None else SKIP
  set_use_fwd_quant(cfg, dlhs_use_fwd_quant, drhs_use_fwd_quant)
  return cfg


def set_scale_and_bias_dtype(cfg: DotGeneral, dtype: jnp.dtype):
  """Set the dtype for all scales and biases in the given DotGeneral config."""
  assert isinstance(
      cfg.fwd.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )
  assert isinstance(
      cfg.dlhs.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )
  assert isinstance(
      cfg.drhs.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )

  def _update_dtype(quantizer: aqt_quantizer.Quantizer):
    calibration_cls = quantizer.calibration
    if calibration_cls is None:
      return

    # TODO(lew): Remove partial inspection wherever possible.
    # Partial inspection is needed because the current implementation of delayed
    # calibration initialization requires the configuration to be set via
    # functools.partial.
    keywords = {}
    if isinstance(calibration_cls, functools.partial):
      keywords = calibration_cls.keywords
      calibration_cls = calibration_cls.func
    keywords.update(dtype=dtype)
    quantizer.calibration = functools.partial(calibration_cls, **keywords)

  _update_dtype(cfg.fwd.dg_quantizer.lhs)
  _update_dtype(cfg.fwd.dg_quantizer.rhs)
  _update_dtype(cfg.dlhs.dg_quantizer.lhs)
  _update_dtype(cfg.dlhs.dg_quantizer.rhs)
  _update_dtype(cfg.drhs.dg_quantizer.lhs)
  _update_dtype(cfg.drhs.dg_quantizer.rhs)


################################################################################
# Functions below are auxiliary config creators.


def default_unquantized_config() -> DotGeneral:
  """Aqt config for floating-point dot general."""

  def tensor_cfg() -> Tensor:
    cfg = Tensor(
        use_fwd_quant=False,
        dequant_mode=DequantMode.OUTPUT,
        calibration_mode=CalibrationMode.CONTRACTING_AXIS,
    )
    return cfg

  def quantizer() -> aqt_quantizer.Quantizer:
    return aqt_quantizer.Quantizer(
        numerics=no_numerics.NoNumerics(),
        calib_shared_axes=None,
        scale_stop_grad=True,
        calibration=None,
        context=utils.Context(key=None, train_step=None),
    )

  def dg_raw_cfg(jax_scope_name: str) -> DotGeneralRaw:
    return DotGeneralRaw(
        lhs=tensor_cfg(),
        rhs=tensor_cfg(),
        dg_quantizer=aqt_dot_general.DefaultDotGeneralQuantizer(
            lhs=quantizer(),
            rhs=quantizer(),
            lhs_mid=quantizer(),
            rhs_mid=quantizer(),
        ),
        dg_accumulator_dtype=None,
        local_aqt=None,
        jax_scope_name=jax_scope_name,
    )

  dg_cfg = DotGeneral(
      fwd=dg_raw_cfg('aqt_fwd'),
      dlhs=dg_raw_cfg('aqt_dlhs'),
      drhs=dg_raw_cfg('aqt_drhs'),
  )
  return dg_cfg


def fully_quantized(
    *,
    fwd_bits: None | int = 8,
    bwd_bits: None | int = 8,
    use_fwd_quant: bool = True,
    use_stochastic_rounding: None | bool = True,
    # Typically we have (but it's a caller's responsibility to check):
    # - vjp_lhs_stochastic_rounding is referring to the gradient and
    # - vjp_rhs_stochastic_rounding is referring to the activations/weights.
    vjp_lhs_stochastic_rounding: None | bool = None,
    vjp_rhs_stochastic_rounding: None | bool = None,
    # The dummy static bound flag is temporary, for performance benchmarking.
    use_dummy_static_bound: bool = False,
    dlhs_local_aqt: None | LocalAqt = None,
    drhs_local_aqt: None | LocalAqt = None,
) -> DotGeneral:
  """Fully Quantized Training."""
  cfg = dot_general_make(
      lhs_bits=fwd_bits,
      rhs_bits=fwd_bits,
      bwd_bits=bwd_bits,
      use_fwd_quant=use_fwd_quant,
      dlhs_local_aqt=dlhs_local_aqt,
      drhs_local_aqt=drhs_local_aqt,
  )

  # Stochastic Rounding
  # These 3 variables are used to ensure we don't mix
  # old and new style of SR configuration.
  old_style_sr_config = use_stochastic_rounding is not None
  new_style_sr_config_lhs = vjp_lhs_stochastic_rounding is not None
  new_style_sr_config_rhs = vjp_rhs_stochastic_rounding is not None
  assert new_style_sr_config_lhs == new_style_sr_config_rhs, (
      'if you use new style SR config (vjp_xhs_stochastic_rounding), do pass'
      ' both lhs and rhs explicitely.'
  )
  assert new_style_sr_config_lhs != old_style_sr_config

  true = True  # A crude way to get around g-explicit-bool-comparison warning

  assert not (vjp_lhs_stochastic_rounding and vjp_rhs_stochastic_rounding), (
      'This config is buggy when you set both to True. Contact lew@ or use'
      ' config_v3'
  )

  # By default use jax.uniform for stochastic rounding
  if use_stochastic_rounding == true:
    set_stochastic_rounding(cfg, True, True, 'jax.uniform')

  if vjp_lhs_stochastic_rounding == true:
    set_stochastic_rounding(cfg, True, False, 'jax.uniform')

  if vjp_rhs_stochastic_rounding == true:
    set_stochastic_rounding(cfg, False, True, 'jax.uniform')

  if use_dummy_static_bound:
    set_constant_calibration(cfg, 1.0)

  assert cfg.fwd.local_aqt is None, 'local_aqt is not yet supported in fwd.'

  return cfg


def config_v3(
    *,
    fwd_bits: None | int = 8,
    dlhs_bits: None | int = 8,
    drhs_bits: None | int = None,
    # The dummy static bound flag is for performance benchmarking.
    use_dummy_static_bound: bool = False,
    rng_type: str = 'jax.uniform',  # 'custom-1'
    dlhs_local_aqt: None | LocalAqt = None,
    drhs_local_aqt: None | LocalAqt = None,
    fwd_accumulator_dtype: ... = jnp.int32,
    dlhs_accumulator_dtype: ... = jnp.int32,
    drhs_accumulator_dtype: ... = None,
) -> DotGeneral:
  """Fully Quantized Training."""
  fwd = dot_general_raw_make(
      fwd_bits,
      fwd_bits,
      jax_scope_name='aqt_fwd',
      initialize_calibration=False,
  )
  dlhs = dot_general_raw_make(
      dlhs_bits,
      dlhs_bits,
      local_aqt=dlhs_local_aqt,
      jax_scope_name='aqt_dlhs',
      initialize_calibration=False,
  )
  drhs = dot_general_raw_make(
      drhs_bits,
      drhs_bits,
      local_aqt=drhs_local_aqt,
      jax_scope_name='aqt_drhs',
      initialize_calibration=False,
  )
  cfg = DotGeneral(fwd=fwd, dlhs=dlhs, drhs=drhs)

  cfg.dlhs.rhs.use_fwd_quant = False
  cfg.drhs.rhs.use_fwd_quant = False

  # Typically we have (but I don't know if it is guaranteed):
  # - vjp_lhs_stochastic_rounding is referring to the gradient and
  # - vjp_rhs_stochastic_rounding is referring to the activations/weights.
  set_stochastic_rounding(
      cfg,
      vjp_lhs_stochastic_rounding=True,
      vjp_rhs_stochastic_rounding=False,
      implementation=rng_type,
  )

  if use_dummy_static_bound:
    set_constant_calibration(cfg, 1.0)

  set_accumulator_dtype(
      cfg,
      fwd_dtype=fwd_accumulator_dtype,
      dlhs_dtype=dlhs_accumulator_dtype,
      drhs_dtype=drhs_accumulator_dtype,
  )
  assert cfg.fwd.local_aqt is None, 'local_aqt is not yet supported in fwd.'
  return cfg


def config_v4(
    *,
    fwd_bits: None | int | fp8_numerics.FP8Dtype = 8,
    dlhs_bits: None | int | fp8_numerics.FP8Dtype = 8,
    drhs_bits: None | int | fp8_numerics.FP8Dtype = None,
    # The dummy static bound flag is for performance benchmarking.
    use_dummy_static_bound: bool = False,
    rng_type: str = 'jax.uniform',  # 'custom-1'
    dlhs_local_aqt: None | LocalAqt = None,
    drhs_local_aqt: None | LocalAqt = None,
    # accumulator dtype by default is automatically set in set_bits,
    # but users can still configure a special dtype such as jnp.int16, etc.
    fwd_accumulator_dtype: None | jnp.dtype | SkipT = SKIP,
    dlhs_accumulator_dtype: None | jnp.dtype | SkipT = SKIP,
    drhs_accumulator_dtype: None | jnp.dtype | SkipT = SKIP,
    dlhs_use_fwd_quant: None | bool | SkipT = SKIP,
    drhs_use_fwd_quant: None | bool | SkipT = SKIP,
    fwd_mid_alpha_both: SkipT | float = SKIP,
    dlhs_mid_alpha_both: SkipT | float = SKIP,
    drhs_mid_alpha_both: SkipT | float = SKIP,
) -> DotGeneral:
  """Version 4 of user-visible AQT config."""
  cfg = default_unquantized_config()
  set_bits(
      cfg,
      fwd_lhs_bit=fwd_bits,
      fwd_rhs_bit=fwd_bits,
      dlhs_lhs_bit=dlhs_bits,
      dlhs_rhs_bit=dlhs_bits,
      drhs_lhs_bit=drhs_bits,
      drhs_rhs_bit=drhs_bits,
  )
  set_accumulator_dtype(
      cfg,
      fwd_dtype=fwd_accumulator_dtype,
      dlhs_dtype=dlhs_accumulator_dtype,
      drhs_dtype=drhs_accumulator_dtype,
  )
  set_stochastic_rounding(
      cfg,
      vjp_lhs_stochastic_rounding=True,
      vjp_rhs_stochastic_rounding=False,
      implementation=rng_type,
  )
  if use_dummy_static_bound:
    set_constant_calibration(cfg, 1.0)
  set_local_aqt(
      cfg,
      fwd_local_aqt=SKIP,
      dlhs_local_aqt=dlhs_local_aqt,
      drhs_local_aqt=drhs_local_aqt,
  )
  set_use_fwd_quant(
      cfg,
      dlhs_use_fwd_quant=dlhs_use_fwd_quant,
      drhs_use_fwd_quant=drhs_use_fwd_quant,
  )
  set_use_mid_quant(
      cfg,
      fwd_mid_alpha_both=fwd_mid_alpha_both,
      dlhs_mid_alpha_both=dlhs_mid_alpha_both,
      drhs_mid_alpha_both=drhs_mid_alpha_both,
  )
  assert cfg.fwd.local_aqt is None, 'local_aqt is not yet supported in fwd.'
  return cfg


def config_fwd_fp8(fwd_bits: fp8_numerics.FP8Dtype = 'e4m3') -> DotGeneral:
  """Configs for FP8 forward pass."""
  assert (
      fwd_bits in fp8_numerics.fp8_map.keys()
  ), 'FP8 only supports 4 or 5 exponent bits'
  cfg = config_v4(fwd_bits=8, dlhs_bits=None, drhs_bits=None)
  set_bits(
      cfg,
      fwd_lhs_bit=fwd_bits,
      fwd_rhs_bit=fwd_bits,
      dlhs_lhs_bit=None,
      dlhs_rhs_bit=None,
      drhs_lhs_bit=None,
      drhs_rhs_bit=None,
  )
  set_stochastic_rounding(cfg, False, False, 'jax.uniform')
  assert cfg.fwd.local_aqt is None, 'local_aqt is not yet supported in fwd.'
  return cfg


def set_fwd_calibration(cfg: DotGeneral, calibration_factory) -> DotGeneral:
  """Updates aqt_cfg for static range calibration."""
  assert isinstance(
      cfg.fwd.dg_quantizer, aqt_dot_general.DefaultDotGeneralQuantizer
  )

  cfg.fwd.dg_quantizer.lhs.calibration = calibration_factory
  cfg.fwd.dg_quantizer.rhs.calibration = calibration_factory
