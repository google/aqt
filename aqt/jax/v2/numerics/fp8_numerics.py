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
"""Numerics for fp8."""

import functools
from typing import Any, Optional
from aqt.jax.v2 import calibration
from aqt.jax.v2 import config
from aqt.jax.v2 import stochastic_rounding
from aqt.jax.v2.flax import aqt_flax
from aqt.jax.v2.numerics import numerics
import flax.struct
import jax
import jax.numpy as jnp


FP8_DTYPE = {'e4m3': jnp.float8_e4m3fn, 'e5m2': jnp.float8_e5m2}


@functools.partial(flax.struct.dataclass, frozen=False, slots=True)
class Fp8Numerics(numerics.AqtNumerics):
  """Numerics for fp8."""

  dtype: Any
  exponent_bits: int = 4
  mantissa_bits: int = 3
  noise_fn: Optional[stochastic_rounding.NoiseFn] = None

  def _get_edge_of_last_fp8_bucket(self):
    return jnp.finfo(self.dtype).max.astype(jnp.bfloat16)

  def get_dtype(self):
    return self.dtype

  def abs_val_mapped_to(self):
    return self._get_edge_of_last_fp8_bucket()

  def vjp_fwd(self, x, context):
    res = (x,)
    if not (
        (self.exponent_bits == 4 and self.mantissa_bits == 3)
        or (self.exponent_bits == 5 and self.mantissa_bits == 2)
    ):
      raise ValueError(
          '(exponent_bits, mantissa_bits) can only be (4,3) or (5,2) but was '
          f'({self.exponent_bits}, {self.mantissa_bits})'
      )

    if self.noise_fn is not None:
      x = (x + self.noise_fn(x.shape, context.key)).astype(x.dtype)

    # clip
    fwd_clip_bound = self._get_edge_of_last_fp8_bucket()
    x = jnp.clip(x, -1 * fwd_clip_bound, fwd_clip_bound)

    # round
    x = round_to_nearest_even(x, self.dtype)

    return x, res

  def vjp_bwd(self, res, grad):
    # This is gradient of clip.
    # For boundary values we will have full gradient.
    # We might use something like this for calibrations other than abs(max(x))
    # (x,) = res
    # ret = (x <= edge_of_last_bucket) * (x >= -edge_of_last_bucket) * grad
    del res
    ret = grad
    return (ret, None)


def tensor_make_fp8(
    exponent_bits: int = 4, mantissa_bits: int = 3
) -> config.Tensor:
  """Makes config.Tensor."""
  # TODO(lew): refactor this into set_numerics()
  return config.Tensor(
      numerics=Fp8Numerics(
          exponent_bits=exponent_bits,
          mantissa_bits=mantissa_bits,
          dtype=FP8_DTYPE[f'e{exponent_bits}m{mantissa_bits}'],
          noise_fn=None,
      ),
      calib_shared_axes=None,
      scale_stop_grad=True,
      calibration=calibration.AbsMaxCalibration(),
      po2_scale=False,
      use_fwd_quant=None,
      context=config.Context(key=None, train_step=None),
      dequant_mode=config.DequantMode.OUTPUT,
  )


def round_to_nearest_even(x: jnp.ndarray, dtype: jnp.dtype) -> jnp.ndarray:
  original_dtype = x.dtype
  x = x.astype(dtype)
  # bitcast_convert to uint8 to avoid allow_excess_precision set in XLA
  x = jax.lax.bitcast_convert_type(x, jnp.uint8)
  x = jax.lax.bitcast_convert_type(x, dtype)
  return x.astype(original_dtype)


def config_fwd_fp8(fwd_bits: str = 'e4m3') -> config.DotGeneral:
  """Configs for FP8 forward pass."""
  assert fwd_bits in FP8_DTYPE.keys(), 'FP8 only supports 4 or 5 exponent bits'
  exponent_bits, mantissa_bits = int(fwd_bits[1]), int(fwd_bits[3])
  cfg = aqt_flax.config_v4(fwd_bits=8, dlhs_bits=None, drhs_bits=None)
  fp8_numerics = Fp8Numerics(
      exponent_bits=exponent_bits,
      mantissa_bits=mantissa_bits,
      dtype=FP8_DTYPE[fwd_bits],
      noise_fn=None,
  )
  config.set_fwd_numerics(cfg, fp8_numerics)
  config.set_accumulator_dtype(cfg, jnp.float32, None, None)
  config.set_stochastic_rounding(cfg, False, False, 'jax.uniform')
  assert cfg.fwd.local_aqt is None, 'local_aqt is not yet supported in fwd.'
  return cfg
