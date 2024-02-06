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

import copy
import enum
from typing import Any, Callable, Optional
from aqt.jax.v2 import calibration
from aqt.jax.v2 import stochastic_rounding
from aqt.jax.v2 import utils
from aqt.jax.v2.numerics import int_numerics
from aqt.jax.v2.numerics import no_numerics
from aqt.jax.v2.numerics import numerics
import jax
import jax.numpy as jnp


DType = Any
AbstractAqtNumerics = numerics.AqtNumerics
AbstractAqtCalibration = calibration.Calibration


@utils.flax_slots_dataclass
class Context:
  key: Optional[jax.Array] = utils.dynamic_field()
  train_step: Optional[int] = utils.dynamic_field()


ClipAndRoundFn = Callable[[jnp.ndarray, Context], jnp.ndarray]

# TODO(lew): move config to aqt_tensor.py and use aqt_tensor.QTensor
QTensor = Any


class DequantMode(enum.Enum):
  """Dequant modes."""

  # Multiply output of dot_general by the transposed scale
  OUTPUT = 1
  # Multiply QTensor.qvalue by untransposed QTensor.scale before
  # dot_general (a.k.a. FakeQuant )
  THIS_INPUT = 2
  # Multiply other argument of dot general by appropriately transposed scale.
  OTHER_INPUT = 3


@utils.flax_slots_dataclass
class Quantizer:
  """Configuration of quantization of one tensor."""
  numerics: AbstractAqtNumerics = utils.static_field()
  calib_shared_axes: Optional[list[int]] = utils.static_field()
  scale_stop_grad: bool = utils.static_field()
  # noise+clip+round
  # We apply gradient of clip_and_round in bwd pass.
  calibration: AbstractAqtCalibration = utils.static_field()
  # Round up the calibration to power of 2 (po2).
  po2_scale: bool = utils.static_field()
  # TODO(yichizh): Factor out auxilliary dataclasses into a separate file.
  context: Context


@utils.flax_slots_dataclass
class Tensor:
  """Configuration of quantization of one tensor or one side of tensor op."""
  quantizer: Quantizer
  # Controls at what value of input tensor should be used.
  # Setting it to True, but not quantizing fwd pass will assert-fail.
  use_fwd_quant: Optional[bool] = utils.static_field()
  # Dequantization mode.
  dequant_mode: DequantMode = utils.static_field()

  @classmethod
  def make(cls, *args, **kwargs) -> 'Tensor':
    return tensor_make(*args, **kwargs)


@utils.flax_slots_dataclass
class LocalAqt:
  contraction_axis_shard_count: int = utils.static_field()


@utils.flax_slots_dataclass
class DotGeneralRaw:
  """Configuration of quantization of one dot_general without gradient."""

  lhs: Tensor
  rhs: Tensor
  dg_accumulator_dtype: Optional[DType] = utils.static_field()
  local_aqt: Optional[LocalAqt] = utils.static_field()
  jax_scope_name: str = utils.static_field()

  @classmethod
  def make(cls, *args, **kwargs) -> 'DotGeneralRaw':
    return dot_general_raw_make(*args, **kwargs)

  @classmethod
  def make_conv_general_dilated(cls, *args, **kwargs) -> 'DotGeneralRaw':
    return conv_general_dilated_make(*args, **kwargs)


@utils.flax_slots_dataclass
class DotGeneral:
  """Configuration of quantization of dot_general and its gradients."""

  fwd: DotGeneralRaw
  dlhs: DotGeneralRaw
  drhs: DotGeneralRaw

  @classmethod
  def make(cls, *args, **kwargs) -> 'DotGeneral':
    return dot_general_make(*args, **kwargs)


################################################################################
# Functions below are auxiliary helpers.


def _split_key(key: Optional[jax.Array], num_splits: int):
  default = (None,) * num_splits
  return default if key is None else jax.random.split(key, num_splits)


def set_context(
    cfg: DotGeneral, key: Optional[jax.Array], train_step: Optional[int]
):
  """Set context with prng keys and train_steps for dot_general config."""
  def set_dg_raw_context(cfg_raw: DotGeneralRaw, key: Optional[jax.Array]):
    key1, key2 = _split_key(key, num_splits=2)
    cfg_raw.lhs.quantizer.context = Context(key=key1, train_step=train_step)
    cfg_raw.rhs.quantizer.context = Context(key=key2, train_step=train_step)

  key_fwd, key_dlhs, key_drhs = _split_key(key, num_splits=3)
  ret_cfg = copy.deepcopy(cfg)
  set_dg_raw_context(ret_cfg.fwd, key_fwd)
  set_dg_raw_context(ret_cfg.dlhs, key_dlhs)
  set_dg_raw_context(ret_cfg.drhs, key_drhs)
  return ret_cfg


def set_fwd_dequant_mode(
    cfg: DotGeneral,
    *,
    lhs_dequant_mode: Optional[DequantMode] = None,
    rhs_dequant_mode: Optional[DequantMode] = None,
):
  if lhs_dequant_mode is not None:
    cfg.fwd.lhs.dequant_mode = lhs_dequant_mode
  if rhs_dequant_mode is not None:
    cfg.fwd.rhs.dequant_mode = rhs_dequant_mode


def set_fwd_numerics(cfg, fwd_numerics: numerics.AqtNumerics):
  cfg.fwd.lhs.quantizer.numerics = fwd_numerics
  cfg.fwd.rhs.quantizer.numerics = fwd_numerics


def set_accumulator_dtype(
    cfg: DotGeneral,
    fwd_dtype: Optional[DType],
    dlhs_dtype: Optional[DType],
    drhs_dtype: Optional[DType],
):
  cfg.fwd.dg_accumulator_dtype = fwd_dtype
  cfg.dlhs.dg_accumulator_dtype = dlhs_dtype
  cfg.drhs.dg_accumulator_dtype = drhs_dtype


def set_stochastic_rounding(
    cfg: DotGeneral,
    # Typically we have (but it's a caller's responsibility to check):
    # - vjp_lhs_stochastic_rounding is referring to the gradient and
    # - vjp_rhs_stochastic_rounding is referring to the activations/weights.
    vjp_lhs_stochastic_rounding: bool,
    vjp_rhs_stochastic_rounding: bool,
    implementation: str,
):
  """Configure stochastic rounding implementation."""
  noise_implementations = {
      'jax.uniform': lambda shape, key: jax.random.uniform(key, shape) - 0.5,
      'custom-1': stochastic_rounding.random_centered_uniform,
  }
  msg = f'{implementation} not supported.'
  assert implementation in noise_implementations.keys(), msg
  noise_fn = noise_implementations[implementation]

  if vjp_lhs_stochastic_rounding:
    cfg.dlhs.lhs.quantizer.numerics = cfg.dlhs.lhs.quantizer.numerics.replace(
        noise_fn=noise_fn
    )
    cfg.drhs.lhs.quantizer.numerics = cfg.drhs.lhs.quantizer.numerics.replace(
        noise_fn=noise_fn
    )
  else:
    cfg.dlhs.lhs.quantizer.numerics = cfg.dlhs.lhs.quantizer.numerics.replace(
        noise_fn=None
    )
    cfg.drhs.lhs.quantizer.numerics = cfg.drhs.lhs.quantizer.numerics.replace(
        noise_fn=None
    )

  if vjp_rhs_stochastic_rounding:
    cfg.dlhs.rhs.quantizer.numerics = cfg.dlhs.rhs.quantizer.numerics.replace(
        noise_fn=noise_fn
    )
    cfg.drhs.rhs.quantizer.numerics = cfg.drhs.rhs.quantizer.numerics.replace(
        noise_fn=noise_fn
    )
  else:
    cfg.dlhs.rhs.quantizer.numerics = cfg.dlhs.rhs.quantizer.numerics.replace(
        noise_fn=None
    )
    cfg.drhs.rhs.quantizer.numerics = cfg.drhs.rhs.quantizer.numerics.replace(
        noise_fn=None
    )


def set_static_bound(cfg: DotGeneral, bound: float = 1.0):
  cfg.fwd.lhs.quantizer.calibration = calibration.ConstantCalibration(bound)
  cfg.fwd.rhs.quantizer.calibration = calibration.ConstantCalibration(bound)
  cfg.drhs.lhs.quantizer.calibration = calibration.ConstantCalibration(bound)
  cfg.drhs.rhs.quantizer.calibration = calibration.ConstantCalibration(bound)
  cfg.dlhs.lhs.quantizer.calibration = calibration.ConstantCalibration(bound)
  cfg.dlhs.rhs.quantizer.calibration = calibration.ConstantCalibration(bound)


def tensor_make(
    bits: Optional[int], preserve_max_val: bool = False
) -> 'Tensor':
  """Makes config.Tensor."""
  if bits is None:
    effective_numerics = no_numerics.NoNumerics()
  else:
    pz = False if bits == 1 else True
    dtype = jnp.int8 if 2 <= bits <= 8 and pz else None
    effective_numerics = int_numerics.IntNumerics(
        bits=bits,
        preserve_zero=pz,
        preserve_max_val=preserve_max_val,
        clip=True,
        round=True,
        noise_fn=None,
        clip_gradient=False,  # This can be disabled when using abs-max scaling.
        dtype=dtype,
    )
  quantizer = Quantizer(
      numerics=effective_numerics,
      calib_shared_axes=None,
      scale_stop_grad=True,
      calibration=calibration.AbsMaxCalibration(),
      po2_scale=False,
      context=Context(key=None, train_step=None),
  )
  return Tensor(
      quantizer=quantizer,
      # dtype_x=dtype,
      use_fwd_quant=None,
      dequant_mode=DequantMode.OUTPUT,
  )


def dot_general_raw_make(
    lhs_bits=None,
    rhs_bits=None,
    local_aqt=None,
    jax_scope_name='aqt',
) -> 'DotGeneralRaw':
  """Create quantization configs for input matrices to a matmul."""
  lhs_cfg = tensor_make(lhs_bits)
  rhs_cfg = tensor_make(rhs_bits)

  # Binary uses 0.5 right now.
  if (
      lhs_bits is not None
      and rhs_bits is not None
      and 2 <= lhs_bits <= 8
      and 2 <= rhs_bits <= 8
  ):
    dg_accumulator_dtype = jnp.int32
  else:
    dg_accumulator_dtype = None

  return DotGeneralRaw(
      lhs=lhs_cfg,
      rhs=rhs_cfg,
      dg_accumulator_dtype=dg_accumulator_dtype,
      local_aqt=local_aqt,
      jax_scope_name=jax_scope_name,
  )


def conv_general_dilated_make(
    spatial_dimensions=2,
    lhs_bits: Optional[int] = None,
    rhs_bits: Optional[int] = None,
) -> 'DotGeneralRaw':
  """Create quantization config conv_general_dilated."""
  config = dot_general_raw_make(lhs_bits, rhs_bits)
  # Hardcoding flax assumptions.
  if config.lhs:
    config.lhs.quantizer.calib_shared_axes = list(
        range(1, spatial_dimensions + 2)
    )
  if config.rhs:
    config.rhs.quantizer.calib_shared_axes = list(
        range(0, spatial_dimensions + 2 - 1)
    )
  return config


def dot_general_make(
    lhs_bits: Optional[int] = None,
    rhs_bits: Optional[int] = None,
    bwd_bits: Optional[int] = None,
    use_fwd_quant: bool = True,
    dlhs_local_aqt=None,
    drhs_local_aqt=None,
) -> 'DotGeneral':
  """Create quantization configs for input matrices to a matmul."""
  fwd = dot_general_raw_make(lhs_bits, rhs_bits, jax_scope_name='aqt_fwd')
  dlhs = dot_general_raw_make(
      bwd_bits, bwd_bits, local_aqt=dlhs_local_aqt, jax_scope_name='aqt_dlhs'
  )
  drhs = dot_general_raw_make(
      bwd_bits, bwd_bits, local_aqt=drhs_local_aqt, jax_scope_name='aqt_drhs'
  )
  cfg = DotGeneral(fwd=fwd, dlhs=dlhs, drhs=drhs)

  # Surprising: lhs quantization determines what drhs can do.
  if lhs_bits is not None:
    # Only rhs is accepting MultiTensor.
    cfg.drhs.rhs.use_fwd_quant = use_fwd_quant
  if rhs_bits is not None:
    cfg.dlhs.rhs.use_fwd_quant = use_fwd_quant
  assert cfg.fwd.local_aqt is None, 'local_aqt is not yet supported in fwd.'
  return cfg


def fully_quantized(
    *,
    fwd_bits: Optional[int] = 8,
    bwd_bits: Optional[int] = 8,
    use_fwd_quant: bool = True,
    use_stochastic_rounding: Optional[bool] = True,
    # Typically we have (but it's a caller's responsibility to check):
    # - vjp_lhs_stochastic_rounding is referring to the gradient and
    # - vjp_rhs_stochastic_rounding is referring to the activations/weights.
    vjp_lhs_stochastic_rounding: Optional[bool] = None,
    vjp_rhs_stochastic_rounding: Optional[bool] = None,
    # The dummy static bound flag is temporary, for performance benchmarking.
    use_dummy_static_bound: bool = False,
    dlhs_local_aqt: Optional[LocalAqt] = None,
    drhs_local_aqt: Optional[LocalAqt] = None,
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
    set_static_bound(cfg, 1.0)

  assert cfg.fwd.local_aqt is None, 'local_aqt is not yet supported in fwd.'

  return cfg


def config_v3(
    *,
    fwd_bits: Optional[int] = 8,
    dlhs_bits: Optional[int] = 8,
    drhs_bits: Optional[int] = None,
    # The dummy static bound flag is for performance benchmarking.
    use_dummy_static_bound: bool = False,
    rng_type: str = 'jax.uniform',  # 'custom-1'
    dlhs_local_aqt: Optional[LocalAqt] = None,
    drhs_local_aqt: Optional[LocalAqt] = None,
    fwd_accumulator_dtype: ... = jnp.int32,
    dlhs_accumulator_dtype: ... = jnp.int32,
    drhs_accumulator_dtype: ... = None,
) -> DotGeneral:
  """Fully Quantized Training."""
  fwd = dot_general_raw_make(fwd_bits, fwd_bits, jax_scope_name='aqt_fwd')
  dlhs = dot_general_raw_make(
      dlhs_bits,
      dlhs_bits,
      local_aqt=dlhs_local_aqt,
      jax_scope_name='aqt_dlhs',
  )
  drhs = dot_general_raw_make(
      drhs_bits,
      drhs_bits,
      local_aqt=drhs_local_aqt,
      jax_scope_name='aqt_drhs',
  )
  cfg = DotGeneral(fwd=fwd, dlhs=dlhs, drhs=drhs)

  cfg.dlhs.rhs.use_fwd_quant = False
  cfg.drhs.rhs.use_fwd_quant = False

  # Typically we have (but I don't know if it is guraranteed):
  # - vjp_lhs_stochastic_rounding is referring to the gradient and
  # - vjp_rhs_stochastic_rounding is referring to the activations/weights.
  set_stochastic_rounding(
      cfg,
      vjp_lhs_stochastic_rounding=True,
      vjp_rhs_stochastic_rounding=False,
      implementation=rng_type,
  )

  if use_dummy_static_bound:
    set_static_bound(cfg, 1.0)

  set_accumulator_dtype(
      cfg,
      fwd_dtype=fwd_accumulator_dtype,
      dlhs_dtype=dlhs_accumulator_dtype,
      drhs_dtype=drhs_accumulator_dtype,
  )
  assert cfg.fwd.local_aqt is None, 'local_aqt is not yet supported in fwd.'
  return cfg
