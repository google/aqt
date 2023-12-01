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

import abc
import dataclasses
from typing import Any, Callable, Optional
from aqt.jax.v2 import calibration
from aqt.jax.v2 import int_numerics
from aqt.jax.v2 import stochastic_rounding
from aqt.jax.v2.numerics import numerics
import flax.struct
import jax
import jax.numpy as jnp

DType = Any
Context = Any  # TODO(lew): We could put Context in a separate file.

ClipAndRoundFn = Callable[[jnp.ndarray, Context], jnp.ndarray]


class Preprocess(abc.ABC):

  @abc.abstractmethod
  def __call__(self, inputs):
    pass


class NoNumerics(numerics.AqtNumerics, flax.struct.PyTreeNode):
  """No quantization, use a native type such as bf16."""

  # TODO(lew): This is a workaround. We should separate Stochastic Rounding.
  # noise_fn has no effect in NoNumerics.
  noise_fn: Optional[stochastic_rounding.NoiseFn] = None

  # TODO(lew): This is a hack. We treat check isinstance(NoNumerics) and treat
  # it in a special way right now. These functions are never called
  def fwd(self, x, context):
    pass

  def abs_val_mapped_to(self):
    pass

  def vjp_fwd(self, x, context):
    pass

  def vjp_bwd(self, res, grad):
    pass


@dataclasses.dataclass
class Tensor:
  """Configuration of quantization of one tensor or one side of tensor op."""

  numerics: numerics.AqtNumerics
  calib_shared_axes: Optional[list[int]]
  scale_stop_grad: bool
  # noise+clip+round
  # We apply gradient of clip_and_round in bwd pass.
  calibration: calibration.Calibration
  # Round up the calibration to power of 2 (po2).
  po2_scale: bool
  use_fake_quant: bool
  # Controls at what value of input tensor should be used.
  # Setting it to True, but not quantizing fwd pass will assert-fail.
  use_fwd_quant: Optional[bool]
  # Operation applied to the quantized inputs to the dot general
  preprocess_quant_cls: Optional[Callable[[], Preprocess]]
  # lax.dot_general output is multiplied by (int transposed) scales.
  # preprocess_scale will be applied to these scales before the multiplication.
  preprocess_scale_cls: Optional[Callable[[], Preprocess]]

  @classmethod
  def make(cls, *args, **kwargs) -> 'Tensor':
    return tensor_make(*args, **kwargs)


@dataclasses.dataclass
class LocalAqt:
  contraction_axis_shard_count: int


@dataclasses.dataclass
class DotGeneralRaw:
  """Configuration of quantization of one dot_general without gradient."""

  lhs: Tensor
  rhs: Tensor
  dg_in_dtype: Optional[DType]
  dg_accumulator_dtype: Optional[DType]
  local_aqt: Optional[LocalAqt]

  @classmethod
  def make(cls, *args, **kwargs) -> 'DotGeneralRaw':
    return dot_general_raw_make(*args, **kwargs)

  @classmethod
  def make_conv_general_dilated(cls, *args, **kwargs) -> 'DotGeneralRaw':
    return conv_general_dilated_make(*args, **kwargs)


@dataclasses.dataclass
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


def set_fwd_numerics(cfg, fwd_numerics: numerics.AqtNumerics):
  cfg.fwd.lhs.numerics = fwd_numerics
  cfg.fwd.rhs.numerics = fwd_numerics


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
    cfg.dlhs.lhs.numerics = cfg.dlhs.lhs.numerics.replace(noise_fn=noise_fn)
    cfg.drhs.lhs.numerics = cfg.drhs.lhs.numerics.replace(noise_fn=noise_fn)
  else:
    cfg.dlhs.lhs.numerics = cfg.dlhs.lhs.numerics.replace(noise_fn=None)
    cfg.drhs.lhs.numerics = cfg.drhs.lhs.numerics.replace(noise_fn=None)

  if vjp_rhs_stochastic_rounding:
    cfg.dlhs.rhs.numerics = cfg.dlhs.rhs.numerics.replace(noise_fn=noise_fn)
    cfg.drhs.rhs.numerics = cfg.drhs.rhs.numerics.replace(noise_fn=noise_fn)
  else:
    cfg.dlhs.rhs.numerics = cfg.dlhs.rhs.numerics.replace(noise_fn=None)
    cfg.drhs.rhs.numerics = cfg.drhs.rhs.numerics.replace(noise_fn=None)


def set_static_bound(cfg: DotGeneral, bound: float = 1.0):
  cfg.fwd.lhs.calibration = calibration.ConstantCalibration(bound)
  cfg.fwd.rhs.calibration = calibration.ConstantCalibration(bound)
  cfg.drhs.lhs.calibration = calibration.ConstantCalibration(bound)
  cfg.drhs.rhs.calibration = calibration.ConstantCalibration(bound)
  cfg.dlhs.lhs.calibration = calibration.ConstantCalibration(bound)
  cfg.dlhs.rhs.calibration = calibration.ConstantCalibration(bound)


def tensor_make(bits: Optional[int]) -> 'Tensor':
  """Makes config.Tensor."""
  if bits is None:
    effective_numerics = NoNumerics()
  else:
    pz = False if bits == 1 else True
    effective_numerics = int_numerics.IntNumerics(
        bits=bits,
        preserve_zero=pz,
        preserve_max_val=False,
        clip=True,
        round=True,
        noise_fn=None,
        clip_gradient=False,  # This can be disabled when using abs-max scaling.
    )

  return Tensor(
      numerics=effective_numerics,
      calib_shared_axes=None,
      scale_stop_grad=True,
      calibration=calibration.AbsMaxCalibration(),
      po2_scale=False,
      use_fake_quant=False,
      # dtype_x=dtype,
      use_fwd_quant=None,
      preprocess_quant_cls=None,
      preprocess_scale_cls=None,
  )


def dot_general_raw_make(
    lhs_bits=None,
    rhs_bits=None,
    local_aqt=None,
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
    dg_in_dtype = jnp.int8
    dg_accumulator_dtype = jnp.int32
  else:
    # Use None to determine the dtype on the fly in aqt_dot_general
    dg_in_dtype = None
    dg_accumulator_dtype = None

  return DotGeneralRaw(
      lhs=lhs_cfg,
      rhs=rhs_cfg,
      dg_in_dtype=dg_in_dtype,
      dg_accumulator_dtype=dg_accumulator_dtype,
      local_aqt=local_aqt,
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
    config.lhs.calib_shared_axes = list(range(1, spatial_dimensions + 2))
  if config.rhs:
    config.rhs.calib_shared_axes = list(range(0, spatial_dimensions + 2 - 1))
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
  fwd = dot_general_raw_make(lhs_bits, rhs_bits)
  dlhs = dot_general_raw_make(bwd_bits, bwd_bits, local_aqt=dlhs_local_aqt)
  drhs = dot_general_raw_make(bwd_bits, bwd_bits, local_aqt=drhs_local_aqt)
  cfg = DotGeneral(fwd=fwd, dlhs=dlhs, drhs=drhs)

  # Surprising: lhs quantization determines what drhs can do.
  if lhs_bits is not None:
    # Only rhs is accepting MultiTensor.
    cfg.drhs.rhs.use_fwd_quant = use_fwd_quant
  if rhs_bits is not None:
    cfg.dlhs.rhs.use_fwd_quant = use_fwd_quant
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

  return cfg


def config_v3(
    *,
    fwd_bits: Optional[int],
    dlhs_bits: Optional[int],
    drhs_bits: Optional[int],
    # The dummy static bound flag is for performance benchmarking.
    use_dummy_static_bound: bool = False,
    rng_type: str = 'jax.uniform',  # 'custom-1'
    dlhs_local_aqt: Optional[LocalAqt] = None,
    drhs_local_aqt: Optional[LocalAqt] = None,
    fwd_accumulator_dtype: ... = jnp.int32,
    dlhs_accumulator_dtype: ... = jnp.int32,
    drhs_accumulator_dtype: ... = jnp.int32,
) -> DotGeneral:
  """Fully Quantized Training."""
  fwd = dot_general_raw_make(fwd_bits, fwd_bits)
  dlhs = dot_general_raw_make(dlhs_bits, dlhs_bits, local_aqt=dlhs_local_aqt)
  drhs = dot_general_raw_make(drhs_bits, drhs_bits, local_aqt=drhs_local_aqt)
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
  return cfg
