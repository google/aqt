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
"""Quantized drop-ins for torch.nn Modules."""

from typing import Optional

from . import functional as QF

from torch import nn, Tensor
import torch.nn.functional as F


class Linear(nn.Linear):

  def __init__(self,
               in_features: int,
               out_features: int,
               bias: bool = True,
               device=None,
               dtype=None,
               input_bits: Optional[int] = None,
               weight_bits: Optional[int] = None) -> None:
    super().__init__(in_features, out_features, bias, device, dtype)
    self.input_bits = input_bits
    self.weight_bits = weight_bits

  def forward(self, input: Tensor) -> Tensor:
    return QF.linear(input, self.weight, self.bias, self.input_bits,
                     self.weight_bits)


class Conv2d(nn.Conv2d):

  def __init__(self,
               in_channels: int,
               out_channels: int,
               kernel_size,
               stride=1,
               padding=0,
               dilation=1,
               groups: int = 1,
               bias: bool = True,
               padding_mode: str = "zeros",
               device=None,
               dtype=None,
               input_bits: Optional[int] = None,
               weight_bits: Optional[int] = None) -> None:
    super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                     dilation, groups, bias, padding_mode, device, dtype)
    self.input_bits = input_bits
    self.weight_bits = weight_bits

  def _conv_forward(
      self,  #
      input: Tensor,
      weight: Tensor,
      bias: Optional[Tensor]) -> Tensor:
    padding = self.padding
    if self.padding_mode != "zeros":
      input = F.pad(input,
                    self._reversed_padding_repeated_twice,
                    mode=self.padding_mode)
      padding = (0, 0)
    return QF.conv2d(input, weight, bias, self.stride, padding, self.dilation,
                     self.groups, self.input_bits, self.weight_bits)
