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
"""Quantized drop-ins for torch.nn.functional."""

from typing import List, Optional

from .quantization import quantize

from torch import Tensor
from torch.nn import functional as F


def linear(input: Tensor,
           weight: Tensor,
           bias: Optional[Tensor] = None,
           input_bits: Optional[int] = None,
           weight_bits: Optional[int] = None):
  # input \in [..., C_in]
  input, input_scale = quantize(input, input_bits, contraction_dims=[-1])
  # weight \in [C_out, C_in]
  weight, weight_scale = quantize(weight, weight_bits, contraction_dims=[-1])
  output = F.linear(input, weight)
  # [..., C_out] * [..., 1]
  output *= input_scale
  # [..., C_out] * [C_out, 1].T
  output *= weight_scale.T
  if bias is not None:
    # [..., C_out] + [C_out]
    output += bias
  return output


def conv2d(input: Tensor,
           weight: Tensor,
           bias: Optional[Tensor] = None,
           stride: List[int] = (1, 1),
           padding: List[int] = (0, 0),
           dilation: List[int] = (1, 1),
           groups: int = 1,
           input_bits: Optional[int] = None,
           weight_bits: Optional[int] = None):
  # input \in [B, C_in, H_in, W_in]
  input, input_scale = quantize(input, input_bits, contraction_dims=[1, 2, 3])
  # weight \in [C_out, C_in/G, K_h, K_w]
  weight, weight_scale = quantize(weight,
                                  weight_bits,
                                  contraction_dims=[1, 2, 3])
  output = F.conv2d(input, weight, None, stride, padding, dilation, groups)
  # [B, C_out, H_out, W_out] * [B, 1, 1, 1]
  output *= input_scale
  # [B, C_out, H_out, W_out] * [C_out, 1, 1, 1].reshape([1, C_out, 1, 1])
  output *= weight_scale.reshape((1, weight.shape[0], 1, 1))
  if bias is not None:
    # [B, C_out, H_out, W_out] + [C_out].reshape([1, C_out, 1, 1])
    output += bias[None, :, None, None]
  return output
