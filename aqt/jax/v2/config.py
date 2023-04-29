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
from typing import Any, Callable, Optional
import jax
import jax.numpy as jnp

DType = Any
NoiseFn = Callable[[tuple[int, ...], jax.random.KeyArray], jnp.ndarray]


@dataclasses.dataclass
class Tensor:
  """Configuration of quantization of one tensor or one side of tensor op."""

  bits: Optional[int]
  calib_shared_axes: Optional[list[int]]
  preserve_zero: bool
  bound: Optional[float]
  bound_stop_grad: bool
  # false = map max val on the end of the last bucket
  # true = map max val on the middle of the last
  preserve_max_val: bool
  clip: bool
  round: bool
  noise_fn: NoiseFn | None
  # Round up the calibration to power of 2 (po2).
  po2_scale: bool

  @classmethod
  def make(cls, bits: Optional[int]) -> 'Tensor':
    pz = False if bits == 1 else True

    return Tensor(
        bits=bits,
        calib_shared_axes=None,
        preserve_zero=pz,
        bound=None,
        bound_stop_grad=True,
        preserve_max_val=False,
        clip=True,
        round=True,
        noise_fn=None,
        po2_scale=False,
    )


@dataclasses.dataclass
class DotGeneralRaw:
  """Configuration of quantization of one dot_general without gradient."""

  lhs: Tensor
  rhs: Tensor
  in_dtype: DType | None
  preferred_element_type: DType
  # use_fwd_quant is observed when this dot_general is used in gradient.
  # use_fwd_quant is ignored in forward pass.
  # Whether the gradient should be taken at unquantized wgt/act or quantized.
  use_fwd_quant: bool
  use_fake_quant: bool

  @classmethod
  def make(cls, lhs_bits=None, rhs_bits=None) -> 'DotGeneralRaw':
    """Create quantization configs for input matrices to a matmul."""
    return DotGeneralRaw(
        lhs=Tensor.make(lhs_bits),
        rhs=Tensor.make(rhs_bits),
        in_dtype=None,
        preferred_element_type=jnp.float32,
        use_fwd_quant=True,
        use_fake_quant=False,
    )

  @classmethod
  def make_conv_general_dilated(
      cls,
      spatial_dimensions=2,
      lhs_bits=None,
      rhs_bits=None,
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
  def make(cls, lhs_bits=None, rhs_bits=None) -> 'DotGeneral':
    """Create quantization configs for input matrices to a matmul."""
    return DotGeneral(
        fwd=DotGeneralRaw.make(lhs_bits, rhs_bits),
        dlhs=DotGeneralRaw.make(),
        drhs=DotGeneralRaw.make(),
    )
