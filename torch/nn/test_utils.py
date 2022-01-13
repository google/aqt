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
"""Test helpers for testing pytorch."""

import unittest

import numpy as np
import torch


def arange_nd(shape, dtype=torch.float32, **tensor_kwargs):
  x = np.arange(np.prod(shape)).reshape(shape)
  return torch.tensor(x, dtype=torch.float32, **tensor_kwargs)


class TorchTestCase(unittest.TestCase):

  def assertAllEqual(self, a, b):
    try:
      torch.testing.assert_allclose(a, b, rtol=0, atol=0)
    except AssertionError as e:
      raise AssertionError(f"\n{a}\n{b}") from e

  def assertAllClose(self, a, b, rtol=None, atol=None):
    try:
      torch.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
    except AssertionError as e:
      raise AssertionError(f"\n{a}\n{b}") from e
