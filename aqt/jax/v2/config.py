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

import dataclasses
from typing import Any, Callable, Optional, Union
from aqt.jax.v2 import calibration
from aqt.jax.v2 import fp8_numerics
from aqt.jax.v2 import int_numerics
from aqt.jax.v2 import stochastic_rounding
import jax
import jax.numpy as jnp

DType = Any
Context = Any  # TODO(lew): We could put Context in a separate file.

ClipAndRoundFn = Callable[[jnp.ndarray, Context], jnp.ndarray]


@dataclasses.dataclass
class NoNumerics:
  """No quantization, use a native type such as bf16."""

  pass


Numerics = Union[NoNumerics, int_numerics.IntNumerics, fp8_numerics.Fp8Numerics]


@dataclasses.dataclass
class Tensor:
  """Configuration of quantization of one tensor or one side of tensor op."""

  numerics: Numerics
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

  @classmethod
  def make(cls, bits: Union[int, Numerics, None]) -> 'Tensor':
    """Makes."""
    if bits is None:
      numerics = NoNumerics()
    elif isinstance(bits, int):
      pz = False if bits == 1 else True
      numerics = int_numerics.IntNumerics(
          bits=bits,
          preserve_zero=pz,
          preserve_max_val=False,
          clip=True,
          round=True,
          noise_fn=None,
      )
    else:
      numerics = bits

    return Tensor(
        numerics=numerics,
        calib_shared_axes=None,
        scale_stop_grad=True,
        calibration=calibration.AbsMaxCalibration(),
        po2_scale=False,
        use_fake_quant=False,
        # dtype_x=dtype,
        use_fwd_quant=None,
    )


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
  def make(
      cls,
      lhs_bits: Union[int, Tensor, None] = None,
      rhs_bits: Union[int, Tensor, None] = None,
      local_aqt: Optional[LocalAqt] = None,
  ) -> 'DotGeneralRaw':
    """Create quantization configs for input matrices to a matmul."""
    lhs_cfg = (
        lhs_bits if isinstance(lhs_bits, Tensor) else Tensor.make(lhs_bits)
    )
    rhs_cfg = (
        rhs_bits if isinstance(rhs_bits, Tensor) else Tensor.make(rhs_bits)
    )

    # Binary uses 0.5 right now.
    if (
        isinstance(lhs_cfg.numerics, int_numerics.IntNumerics)
        and isinstance(rhs_cfg.numerics, int_numerics.IntNumerics)
        and 2 <= lhs_cfg.numerics.bits <= 8
        and 2 <= rhs_cfg.numerics.bits <= 8
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

  @classmethod
  def make_conv_general_dilated(
      cls,
      spatial_dimensions=2,
      lhs_bits: Union[int, Tensor, None] = None,
      rhs_bits: Union[int, Tensor, None] = None,
  ) -> 'DotGeneralRaw':
    """Create quantization config conv_general_dilated."""
    config = cls.make(lhs_bits, rhs_bits)
    # Hardcoding flax assumptions.
    if config.lhs:
      config.lhs.calib_shared_axes = list(range(1, spatial_dimensions + 2))
    if config.rhs:
      config.rhs.calib_shared_axes = list(range(0, spatial_dimensions + 2 - 1))
    return config


@dataclasses.dataclass
class DotGeneral:
  """Configuration of quantization of dot_general and its gradients."""

  fwd: DotGeneralRaw
  dlhs: DotGeneralRaw
  drhs: DotGeneralRaw

  @classmethod
  def make(
      cls,
      lhs_bits: Union[int, Tensor, None] = None,
      rhs_bits: Union[int, Tensor, None] = None,
      bwd_bits: Union[int, Tensor, None] = None,
      use_fwd_quant: bool = True,
      dlhs_local_aqt=None,
      drhs_local_aqt=None,
  ) -> 'DotGeneral':
    """Create quantization configs for input matrices to a matmul."""
    fwd = DotGeneralRaw.make(lhs_bits, rhs_bits)
    dlhs = DotGeneralRaw.make(bwd_bits, bwd_bits, local_aqt=dlhs_local_aqt)
    drhs = DotGeneralRaw.make(bwd_bits, bwd_bits, local_aqt=drhs_local_aqt)
    cfg = cls(fwd=fwd, dlhs=dlhs, drhs=drhs)

    # Surprising: lhs quantization determines what drhs can do.
    if lhs_bits is not None:
      # Only rhs is accepting MultiTensor.
      cfg.drhs.rhs.use_fwd_quant = use_fwd_quant
    if rhs_bits is not None:
      cfg.dlhs.rhs.use_fwd_quant = use_fwd_quant
    return cfg


def fully_quantized(
    *,
    fwd_bits: Union[int, Tensor, None] = 8,
    bwd_bits: Union[int, Tensor, None] = 8,
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
  cfg = DotGeneral.make(
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


def set_accumulator_dtype(
    cfg: DotGeneral,
    fwd_dtype: Optional[DType],
    bwd_dtype: Optional[DType],
):
  cfg.fwd.dg_accumulator_dtype = fwd_dtype
  cfg.dlhs.dg_accumulator_dtype = bwd_dtype
  cfg.drhs.dg_accumulator_dtype = bwd_dtype


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
    cfg.dlhs.lhs.noise_fn = noise_fn
    cfg.drhs.lhs.noise_fn = noise_fn
  else:
    cfg.dlhs.lhs.noise_fn = None
    cfg.drhs.lhs.noise_fn = None

  if vjp_rhs_stochastic_rounding:
    cfg.dlhs.rhs.noise_fn = noise_fn
    cfg.drhs.rhs.noise_fn = noise_fn
  else:
    cfg.dlhs.rhs.noise_fn = None
    cfg.drhs.rhs.noise_fn = None


def set_static_bound(cfg: DotGeneral, bound: float = 1.0):
  cfg.fwd.lhs.calibration = calibration.ConstantCalibration(bound)
  cfg.fwd.rhs.calibration = calibration.ConstantCalibration(bound)
  cfg.drhs.lhs.calibration = calibration.ConstantCalibration(bound)
  cfg.drhs.rhs.calibration = calibration.ConstantCalibration(bound)
  cfg.dlhs.lhs.calibration = calibration.ConstantCalibration(bound)
  cfg.dlhs.rhs.calibration = calibration.ConstantCalibration(bound)


def int8_ttf_quant_v1(use_stochastic_rounding=True) -> DotGeneral:
  """Version 1 of 'TTF' int8 quantized training recipe."""
  fwd = DotGeneralRaw.make(8, 8)
  dlhs = DotGeneralRaw.make(8, 8)
  drhs = DotGeneralRaw.make(None, None)
  cfg = DotGeneral(fwd=fwd, dlhs=dlhs, drhs=drhs)

  # Surprising: lhs quantization determines what drhs can do.
  # Only rhs is accepting MultiTensor.
  cfg.drhs.rhs.use_fwd_quant = False
  cfg.dlhs.rhs.use_fwd_quant = False
  if use_stochastic_rounding:
    set_stochastic_rounding(
        cfg,
        vjp_lhs_stochastic_rounding=True,
        vjp_rhs_stochastic_rounding=False,
        implementation='jax.uniform',
    )

  return cfg
