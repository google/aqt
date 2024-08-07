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

from aqt.jax.v2 import utils
from aqt.jax.v2.numerics import numerics
import jax
import jax.numpy as jnp


bit_repr = jax.Array
float_repr = jax.Array


@utils.flax_slots_kw_only_dataclass
class FpNumericsConfig:
  nexp: int
  minexp: int
  nmant: int
  has_subnormals: bool
  has_two_nan: bool
  has_naninf: bool

e3m0 = FpNumericsConfig(
    nexp=3,
    minexp=0,
    nmant=0,
    has_subnormals=False,
    has_two_nan=False,
    has_naninf=False,
)

e2m1 = FpNumericsConfig(
    nexp=2,
    minexp=0,
    nmant=1,
    has_subnormals=False,
    has_two_nan=False,
    has_naninf=False,
)

e1m0 = FpNumericsConfig(
    nexp=1,
    minexp=0,
    nmant=0,
    has_subnormals=False,
    has_two_nan=False,
    has_naninf=False,
)

float8_e4m3fn = FpNumericsConfig(
    nexp=4,
    minexp=-6,
    nmant=3,
    has_subnormals=True,
    has_two_nan=True,
    has_naninf=False,
)

float8_e5m2 = FpNumericsConfig(
    nexp=5,
    minexp=-14,
    nmant=2,
    has_subnormals=True,
    has_two_nan=False,
    has_naninf=True,
)

float16 = FpNumericsConfig(
    nexp=5,
    minexp=-14,
    nmant=10,
    has_subnormals=True,
    has_two_nan=False,
    has_naninf=True,
)


# TODO(lew): Add minexp.
def fp_round(
    x,
    *,
    cfg: FpNumericsConfig,
    key: jax.Array,
    stochastic_rounding: bool,
    test_noise_axis=None,
):
  """FP stochastic rounding for a given mantissa and exponent. Returns bf16."""
  # Information about the numerics to be simulated
  nexp = cfg.nexp
  nmant = cfg.nmant
  assert cfg.minexp == 0, 'minexp not implemented'
  assert not cfg.has_two_nan, 'two_nan not implemented'
  assert not cfg.has_naninf, 'naninf not implemented'

  # Information about the container
  # BF16 bit pattern: (sign, exp, mantissa) = (1,8,x)
  # Notation: ctner = container
  input_dtype = x.dtype
  msg = f'Unsupported dtype for sthochastic rounding: {input_dtype}.'
  assert input_dtype == jnp.bfloat16, msg
  ctner_nmant = jnp.finfo(input_dtype).nmant
  assert nmant <= ctner_nmant, 'bf16 man bits'
  bits_dtype = jnp.uint16

  # Masks
  ctner_mant_mask = bits_dtype((1 << ctner_nmant) - 1)  # 0_00000000_1111111
  mant_trunc_bits = ctner_nmant - nmant
  # Mantissa truncation mask:
  #     exp    nmant
  # 0_00000000_0..0011
  mant_trunc_mask = bits_dtype((1 << mant_trunc_bits) - 1)

  min_normal = jax.lax.convert_element_type(2**cfg.minexp, input_dtype)
  bias = 1 if cfg.has_subnormals else 0
  ctner_bias = 127
  min_ctner_exp = bits_dtype(ctner_bias - bias)
  max_ctner_exp = bits_dtype(min_ctner_exp + 2**nexp - 1)
  max_ctner_mant = bits_dtype(ctner_mant_mask & ~mant_trunc_mask)
  max_normal_repr = (max_ctner_exp << ctner_nmant) + max_ctner_mant
  max_normal = jax.lax.bitcast_convert_type(max_normal_repr, input_dtype)

  def trunc_and_clip_normal(x_in: float_repr) -> float_repr:
    sign = jnp.sign(x_in)
    x_in = jnp.abs(x_in)
    x_in = jax.lax.bitcast_convert_type(x_in, bits_dtype)
    if stochastic_rounding:
      if test_noise_axis is not None:
        noise_axis_size = x_in.shape[test_noise_axis]
        sh = list((1,) * len(x_in.shape))
        sh[test_noise_axis] = noise_axis_size
        rnd_bits = jnp.arange(noise_axis_size, dtype=bits_dtype).reshape(sh)
      else:
        rnd_bits = jax.random.bits(key, x_in.shape, bits_dtype)
      noise = jax.lax.bitwise_and(rnd_bits, mant_trunc_mask)
      x_in = x_in + noise
    else:
      carry_bit = bits_dtype(1 << (mant_trunc_bits - 1))
      x_in = x_in + carry_bit
    x_in = jax.lax.bitwise_and(x_in, ~mant_trunc_mask)
    x_in = jax.lax.bitcast_convert_type(x_in, input_dtype)
    if nexp is not None:
      x_in = jnp.clip(x_in, min_normal, max_normal)
    return x_in * sign

  def round_subnormal(x_in: float_repr) -> float_repr:
    if cfg.has_subnormals:
      if stochastic_rounding:
        if test_noise_axis is not None:
          noise_axis_size = x_in.shape[test_noise_axis]
          sh = list((1,) * len(x_in.shape))
          sh[test_noise_axis] = noise_axis_size
          rnd_bits = jnp.arange(noise_axis_size, dtype=bits_dtype).reshape(sh)
          rnd_bits = rnd_bits << 8
          rnd_bits = (jnp.uint32(rnd_bits) << 7) | jnp.float32(1).view(
              jnp.uint32
          )
          noise = (
              jax.lax.bitcast_convert_type(rnd_bits, jnp.float32)
              - 1.5
              + 0.001953125  # to make noise symmetric to zero
          )
        else:
          noise = jax.random.uniform(key, x_in.shape, dtype=jnp.float32) - 0.5
      else:
        noise = 0.0
      x_in = (jnp.round(x_in * 2**nmant + noise) / 2**nmant).astype(input_dtype)
    else:
      if stochastic_rounding:
        if test_noise_axis is not None:
          noise_axis_size = x_in.shape[test_noise_axis]
          sh = list((1,) * len(x_in.shape))
          sh[test_noise_axis] = noise_axis_size
          rnd_bits = jnp.arange(noise_axis_size, dtype=bits_dtype).reshape(sh)
          rnd_bits = rnd_bits << 8
        else:
          rnd_bits = jax.random.bits(key, x_in.shape, bits_dtype)
        rnd_bits = (jnp.uint32(rnd_bits) << 7) | jnp.float32(2).view(jnp.uint32)
        rnd_floats = jax.lax.bitcast_convert_type(rnd_bits, jnp.float32) - (
            3.0 - 2 ** (-16)
        )
        x_in = (x_in > rnd_floats) * 2 - 1.0
      else:
        x_in = jnp.sign(x_in).astype(input_dtype)
    return x_in

  normal = trunc_and_clip_normal(x)
  subnormal = round_subnormal(x)
  out = jnp.where(jnp.abs(x) < min_normal, subnormal, normal)
  return out


def fp_largest_representable(cfg: FpNumericsConfig):
  nexp = cfg.nexp
  nmant = cfg.nmant
  assert cfg.minexp == 0, 'minexp not implemented'
  assert not cfg.has_naninf, 'naninf not implemented'
  max_exp = 2**nexp - 1 + cfg.minexp - (1 if cfg.has_subnormals else 0)
  max_mant = 2**nmant - 1 - (1 if cfg.has_two_nan else 0)
  return 2**max_exp * (1 + (max_mant / 2**nmant))


@utils.flax_slots_kw_only_dataclass
class FpNumerics(numerics.AqtNumerics):
  """Numerics for fp8."""

  # Requested rounding precision.
  cfg: FpNumericsConfig = utils.static_field()
  stochastic_rounding: bool = utils.static_field(default=False)
  clip_gradient: bool = utils.static_field(default=False)

  def abs_val_mapped_to(self):
    return fp_largest_representable(cfg=self.cfg)

  def get_dtype(self):
    return jnp.bfloat16

  def vjp_fwd(self, x, context):
    x = fp_round(
        x.astype(jnp.bfloat16),
        cfg=self.cfg,
        key=context.key,
        stochastic_rounding=self.stochastic_rounding,
    )
    res = (x,)
    return x, res

  def vjp_bwd(self, res, grad):
    ret = grad
    if self.clip_gradient:
      (x,) = res
      clip_bound = self.abs_val_mapped_to()
      ret *= (-clip_bound <= x) * (x <= clip_bound)
    return (ret, None)
