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
"""Tests for quantization.py."""
# Run using `python -m torch.nn.quantization_test`

import unittest

from .quantization import quantize, get_max_signed_int, get_max_unsigned_int
from .test_utils import arange_nd, TorchTestCase

import torch


class QuantizationTest(TorchTestCase):

  def test_bits_none(self):
    x = arange_nd((10, 4))
    x_quant, x_scale = quantize(x, bits=None, contraction_dims=[1])
    self.assertAllEqual(x, x_quant)
    self.assertAllEqual(x_scale, torch.ones([10, 1]))

  def test_full_range_int_inputs_quantization(self):
    # Integer weights in full range [min_val, max_value] quantizes correctly.
    for is_positive in [False, True]:
      for bits in [2, 4, 8, 16, 24, 32]:
        if not is_positive:
          max_val = int(get_max_signed_int(bits))
          min_val = -max_val
        else:
          min_val = 0
          max_val = int(get_max_unsigned_int(bits))
        x = torch.randint(min_val, max_val, (10, 1)).type(torch.float32)
        # Explicitly set min and max vals to test the full range and ensure
        # symmetric quantization is applied if it is being tested.
        x[0] = min_val
        x[-1] = max_val

        with self.subTest(f"bits={bits}, is_positive={is_positive}"):
          x_quant, x_scale = quantize(x, bits, contraction_dims=[])
          self.assertAllEqual(x, x_quant)
          self.assertAllEqual(x_scale, torch.ones_like(x_scale, dtype=x.dtype))

  def test_per_feature_scale_invariance_inputs_quantization(self):
    for is_positive in [False, True]:
      for bits in [2, 4, 8, 16, 24, 32]:
        # Tests that dequant(quant(x * 2)) == dequant(quant(x)) * 2
        if is_positive:
          x = torch.rand((10, 4))
        else:
          x = torch.randn((10, 4))
        y_scale = 2**torch.arange(4)
        y = x * y_scale

        x_quant, x_quant_scale = quantize(x, bits, contraction_dims=[0])
        x_dequant = x_quant * x_quant_scale

        y_quant, y_quant_scale = quantize(y, bits, contraction_dims=[0])
        y_dequant = y_quant * y_quant_scale

        self.assertAllClose(x_dequant * y_scale, y_dequant)


if __name__ == "__main__":
  unittest.main()
