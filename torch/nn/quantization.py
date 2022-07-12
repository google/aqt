# Copyright 2021 Google LLC
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
"""Utilities for dynamic max-value quantization."""

from typing import List, Optional, Tuple

import torch
from torch import Tensor


def _get_integral_float_less_than_power_of_two(exp: int) -> float:
  """Returns the largest float32 integer value smaller than 2**exp."""
  if exp <= 24:
    # 2 ** exp - 1 can be represented exactly.
    return 2.**exp - 1
  else:
    # Starting with 2**24, each binary order of magnitude has float32 values
    # spaced 2, 4, 8,... integer values apart, and for exp > 24, 2**exp - 1
    # is rounded to 2**exp. We avoid casting or overflow errors by computing the
    # largest float value smaller than 2**exp explicitly.
    return 2.**exp - 2.**(exp - 24)


def get_max_signed_int(bits: int) -> float:
  """Compute the max value for a signed int with the given number of bits."""
  if bits <= 1:
    raise ValueError("quantization is only supported for 2 or more bits")
  return _get_integral_float_less_than_power_of_two(bits - 1)


def get_max_unsigned_int(bits: int) -> float:
  """Compute the max value for an unsigned int with the given number of bits."""
  if bits <= 1:
    raise ValueError("quantization is only supported for 2 or more bits")
  return _get_integral_float_less_than_power_of_two(bits)


def scale_max_to_signed_range(
    x: Tensor,  #
    bits: int,
    dims: List[int]) -> Tuple[Tensor, Tensor]:
  """Scales x to the range of a signed int with the given number of bits.

  -2**(bits - 1) + 1 <= x_scaled <= 2**(bits - 1) - 1.

  Gradients are not propagated for the scale.

  Args:
    x: The input to compute the scale for.
    bits: The number of bits to quantize x to.
    dims: Dimensions of input to consider for scaling.

  Returns:
    x scaled such that its largest value is equal to MAX_INT<bits> and the scale
    x was divided by. This scale has the same shape as x, and can be used to
    rescale the output of a linear operation that x was an input to.
  """
  max_val = torch.amax(torch.abs(x), dim=dims, keepdim=True)
  scale = max_val / get_max_signed_int(bits)
  scale = scale.detach()
  # The scale will be zero if all values in a channel of x are zero. We use
  # nan_to_num to ensure that x / scale * scale == x for this case.
  x = torch.nan_to_num_(x / scale)
  return x, scale


def scale_max_to_unsigned_range(
    x: Tensor,  #
    bits: int,
    dims: List[int]) -> Tuple[Tensor, Tensor]:
  """Scales x to the range of an unsigned int with the given number of bits.

  0 <= x_scaled <= 2**bits, where min(x) >= 0.

  Gradients are not propagated for the scale.

  Args:
    x: The input to compute the scale for.
    bits: The number of bits to quantize x to.
    dims: Dimensions of input to consider for scaling.

  Returns:
    x scaled such that its largest value is equal to MAX_UINT<bits> and the
    scale x was divided by. This scale has the same shape as x, and can be used
    to rescale the output of a linear operation that x was an input to.
  """
  max_val = torch.amax(x, dim=dims, keepdim=True)
  scale = max_val / get_max_unsigned_int(bits)
  scale = scale.detach()
  # The scale will be zero if all values in a channel of x are zero. We use
  # nan_to_num to ensure that x / scale * scale == x for this case.
  x = torch.nan_to_num_(x / scale)
  return x, scale


def ste(forward):
  """Returns 'forward' with the straight through estimator as its gradient."""

  class StraightThroughEstimator(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
      return forward(input)

    @staticmethod
    def backward(ctx, grad_output):
      return grad_output

  return StraightThroughEstimator.apply


def integerize_to_signed_range(x: Tensor, bits: int) -> Tensor:
  """Round x and clamp it the signed int range for the given number of bits.

  Args:
    x: The input tensor. Output of scale_max_to_signed_range when quantizing.
    bits: The number of bits to quantize x to.

  Returns:
    x rounded and clamped to the appropriate range.
  """
  x = ste(torch.round)(x)
  max_int = get_max_signed_int(bits)
  return torch.clamp(x, -max_int, max_int)


def integerize_to_unsigned_range(x: Tensor, bits: int) -> Tensor:
  """Floor x and clamp it the unsigned int range for the given number of bits.

  Args:
    x: The input tensor. Output of scale_max_to_unsigned_range when quantizing.
    bits: The number of bits to quantize x to.

  Returns:
    x floored and clamped to the appropriate range.
  """
  x = ste(torch.round)(x)
  return torch.clamp(x, 0, get_max_unsigned_int(bits))


def quantize(
    x: Tensor,  #
    bits: Optional[int],
    contraction_dims: List[int]) -> Tuple[Tensor, Optional[Tensor]]:
  """Quantize x dynamically, returning the quantized value and scale.

  The output should be multiplied by the returned scale after the subsequent
  linear operation (e.g. matmul, einsum, conv, ...).

  x, x_scale = quantize(x, bits, dims=[1])
  w, w_scale = quantize(w, bits, dims=[0])
  y = matmul(x, w)
  y = y * x_scale * w_scale

  The scale returned has the same shape as x but with 1 in place of each
  contraction dim. Depending on the op the scale may need to be reshaped before
  multiplying it with the output.

  Args:
    x: The tensor to quantize.
    bits: The number of bits to quantize x to..
    contraction_dims: The dimensions along which to quantize (complement of the
      feature dims).

  Returns:
    Quantized value and scale; the scale has the same rank as x, as we retain
    reduced dimensions as 1. For the bits=None case, we return x without
    quantizing and a ones tensor for the scale.
  """
  if bits is None:
    # Normalize dims to all be positive.
    contraction_dims = [
        d if d >= 0 else len(x.shape) + d for d in contraction_dims
    ]
    # Replace all contraction dims with one.
    scale_shape = [
        1 if i in contraction_dims else dim for i, dim in enumerate(x.shape)
    ]
    return x, torch.ones(scale_shape, dtype=x.dtype)
  if (x < 0).any():
    x, scale = scale_max_to_signed_range(x, bits, contraction_dims)
    x = integerize_to_signed_range(x, bits)
  else:
    x, scale = scale_max_to_unsigned_range(x, bits, contraction_dims)
    x = integerize_to_unsigned_range(x, bits)
  return x, scale
