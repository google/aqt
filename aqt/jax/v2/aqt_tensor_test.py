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

"""Test for AQT tensor."""
from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax.v2 import aqt_tensor
import jax.numpy as jnp


class AqtTensorTest(parameterized.TestCase):

  def test_dynamic_slice(self):
    x = jnp.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [3, 5, 1, 10],
    ])
    scale = jnp.max(jnp.abs(x), axis=1, keepdims=True)
    print(x)
    print(scale)
    print(x.shape, scale.shape)

    q = aqt_tensor.QTensor(qvalue=x, scale=[scale], scale_t=None)
    y = aqt_tensor.dynamic_slice(q, start_indices=(1, 0), slice_sizes=[2, 1])
    print("======")
    print(y.qvalue)
    print(y.scale[0])

  def test_getitem(self):
    x = jnp.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [3, 5, 1, 10],
    ])
    scale = jnp.max(jnp.abs(x), axis=0, keepdims=True)
    print(x)
    print(scale)
    print(x.shape, scale.shape)

    q = aqt_tensor.QTensor(qvalue=x, scale=[scale], scale_t=None)
    y = q.__getitem__(2)
    print("======")
    print(y.qvalue)
    print(y.scale[0])


if __name__ == "__main__":
  absltest.main()
