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
"""Util functions for numerics in AQT v2."""

from aqt.jax.v2 import utils
from aqt.jax.v2.numerics import fp8_numerics
from aqt.jax.v2.numerics import int_numerics
from aqt.jax.v2.numerics import no_numerics


def get_numerics(
    bits: None | int | fp8_numerics.FP8Dtype, preserve_max_val=False
):
  """Get numerics object from number of bits."""
  if bits is None:
    effective_numerics = no_numerics.NoNumerics()
  elif bits in fp8_numerics.fp8_map.keys():
    exponent_bits, mantissa_bits = int(bits[1]), int(bits[3])
    effective_numerics = fp8_numerics.Fp8Numerics(
        exponent_bits=exponent_bits,
        mantissa_bits=mantissa_bits,
        dtype=fp8_numerics.fp8_map[bits],
    )
  else:
    pz = False if bits == 1 else True
    dtype = utils.infer_dtype_from_bits(bits) if pz else None
    effective_numerics = int_numerics.IntSymmetric(
        bits=bits,
        preserve_zero=pz,
        preserve_max_val=preserve_max_val,
        clip=True,
        round=True,
        noise_fn=None,
        clip_gradient=False,  # Can be disabled when using abs-max scaling.
        dtype=dtype,
    )
  return effective_numerics
