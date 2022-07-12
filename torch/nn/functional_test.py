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
"""Tests for functional.py."""
# Run using `python -m torch.nn.functional_test`

import unittest

from . import functional as QF
from .test_utils import arange_nd, TorchTestCase

import torch
import torch.nn.functional as F


class FunctionalTest(TorchTestCase):

  def test_linear_close_for_high_bit_depths(self):
    x = arange_nd((2, 3, 4, 5), requires_grad=True)
    w = arange_nd((6, 5))
    bias = torch.arange(6, dtype=torch.float32)

    # Compare output against torch implementation.
    y = F.linear(x, w, bias)
    y_quant = QF.linear(x, w, bias, input_bits=16, weight_bits=16)
    self.assertAllClose(y, y_quant)

    # Compute the expected gradient.
    torch.sum(y).backward()
    x_grad = torch.clone(x.grad)
    x.grad.data = torch.zeros_like(x.grad.data)

    # Compute the quantized gradient and comapre.
    torch.sum(y_quant).backward()
    x_quant_grad = x.grad
    self.assertAllClose(x_grad, x_quant_grad)

  def test_linear_none_quant(self):
    x = arange_nd((2, 3, 4, 5), requires_grad=True)
    w = arange_nd((6, 5))
    bias = torch.arange(6, dtype=torch.float32)

    # Compare output against torch implementation.
    y = F.linear(x, w, bias)
    y_quant = QF.linear(x, w, bias)
    self.assertAllEqual(y, y_quant)

  def test_conv2d_close_for_high_bit_depths(self):
    for stride in [[1, 1], [1, 2]]:
      for padding in [[0, 0], [2, 3]]:
        for dilation in [1, 2]:
          for groups in [1, 2]:
            x = arange_nd((2, 3 * groups, 10, 13), requires_grad=True)
            w = arange_nd((4, 3, 5, 5))
            bias = torch.arange(4, dtype=torch.float32)

            # Use -x to test the symmetric quantization path.
            y = F.conv2d(-x, w, bias, stride, padding, dilation, groups)
            y_quant = QF.conv2d(-x,
                                w,
                                bias,
                                stride,
                                padding,
                                dilation,
                                groups,
                                input_bits=16,
                                weight_bits=16)
            self.assertAllClose(y, y_quant)

            # Compute the expected gradient.
            torch.sum(y).backward()
            x_grad = torch.clone(x.grad)
            x.grad.data = torch.zeros_like(x.grad.data)

            # Compute the quantized gradient and comapre.
            torch.sum(y_quant).backward()
            x_quant_grad = x.grad
            self.assertAllClose(x_grad, x_quant_grad)

  def test_conv_none_quant(self):
    for stride in [[1, 1], [1, 2]]:
      for padding in [[0, 0], [2, 3]]:
        for dilation in [1, 2]:
          for groups in [1, 2]:
            x = arange_nd((2, 3 * groups, 10, 13))
            w = arange_nd((4, 3, 5, 5))
            bias = torch.arange(4, dtype=torch.float32)

            y = F.conv2d(x, w, bias, stride, padding, dilation, groups)
            y_quant = QF.conv2d(x,
                                w,
                                bias,
                                stride,
                                padding,
                                dilation,
                                groups,
                                input_bits=None,
                                weight_bits=None)

            self.assertAllEqual(y, y_quant)


if __name__ == "__main__":
  unittest.main()
