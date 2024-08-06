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
from aqt.jax.v2 import tiled_dot_general
import jax
import jax.numpy as jnp


TensorTiling = tiled_dot_general.TensorTiling
AxisTiling = tiled_dot_general.AxisTiling


# TODO(yichizh): refactor and clean the test cases
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

    q = aqt_tensor.QTensor(
        qvalue=x,
        scale=[scale],
        scale_t=None,
        bias=[],
        dequant_dtype=scale.dtype,
    )
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

    q = aqt_tensor.QTensor(
        qvalue=x,
        scale=[scale],
        scale_t=None,
        bias=[],
        dequant_dtype=scale.dtype,
    )
    y = q.__getitem__(2)
    print("======")
    print(y.qvalue)
    print(y.scale[0])

  def test_dynamic_update(self):
    x = jnp.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [3, 5, 1, 10],
    ])
    scale = jnp.max(jnp.abs(x), axis=0, keepdims=True)
    print(x)
    print(scale)
    print(x.shape, scale.shape)
    q = aqt_tensor.QTensor(
        qvalue=x,
        scale=[scale],
        scale_t=None,
        bias=[],
        dequant_dtype=scale.dtype,
    )

    update_qvalue = jnp.zeros((3, 1), dtype=x.dtype)
    update_scale = jnp.max(jnp.abs(update_qvalue), axis=0, keepdims=True)
    update = aqt_tensor.QTensor(
        qvalue=update_qvalue,
        scale=[update_scale],
        scale_t=None,
        bias=[],
        dequant_dtype=update_scale.dtype,
    )
    y = aqt_tensor.dynamic_update_slice(q, update, (0, 1))
    print("======")
    print(y.qvalue)
    print(y.scale[0])

  def test_dtype(self):
    qt = aqt_tensor.zeros(
        shape=(1,), container_dtype=jnp.int8, dequant_dtype=jnp.float32
    )
    self.assertEqual(qt.dtype, jnp.float32)
    self.assertIsInstance(qt.dtype, jnp.dtype)
    self.assertEqual(qt.dequant_dtype, jnp.float32)

    qt = qt.astype(jnp.bfloat16)
    self.assertEqual(qt.dtype, jnp.bfloat16)
    self.assertIsInstance(qt.dtype, jnp.dtype)
    self.assertEqual(qt.dequant_dtype, jnp.bfloat16)

  def test_tiling_state(self):
    lhs_key, rhs_key = jax.random.split(jax.random.PRNGKey(0))
    lhs = jax.random.randint(lhs_key, shape=(30, 40, 50), minval=0, maxval=100)
    rhs = jax.random.randint(rhs_key, shape=(40, 50, 60), minval=0, maxval=100)

    cfg = tiled_dot_general.Cfg(
        lhs=TensorTiling(
            contraction_axes=[
                AxisTiling(axis=1, tile_size=10, tile_count=None),
                AxisTiling(axis=2, tile_size=25, tile_count=None),
            ],
            remaining_axes=[],
        ),
        rhs=TensorTiling(
            contraction_axes=[
                AxisTiling(axis=0, tile_size=10, tile_count=None),
                AxisTiling(axis=1, tile_size=25, tile_count=None),
            ],
            remaining_axes=[],
        ),
    )

    # make tiling state for both lhs and rhs
    lhs_ca = (1, 2)
    rhs_ca = (0, 1)
    xlhs, xrhs = tiled_dot_general.generate_tiling_states_for_dot_general(
        cfg, lhs, rhs, dimension_numbers=((lhs_ca, rhs_ca), ((), ()))
    )

    # make scale
    _, xlhs_ca = xlhs.to_tiled_axes_transposed(lhs_ca)
    xlhs_scale = xlhs.apply(lhs)
    xlhs_scale = jnp.max(xlhs_scale, axis=xlhs_ca, keepdims=True) / (2.0**7 - 1)

    _, xrhs_ca = xrhs.to_tiled_axes_transposed(rhs_ca)
    xrhs_scale = xrhs.apply(rhs)
    xrhs_scale = jnp.max(xrhs_scale, axis=xrhs_ca, keepdims=True) / (2.0**7 - 1)

    # make qtensor
    qlhs = aqt_tensor.QTensor(
        qvalue=None,
        scale=[xlhs_scale],
        scale_t=None,
        bias=[],
        dequant_dtype=xlhs_scale.dtype,
        tiling_state=xlhs,
    )
    qlhs = qlhs.quant(lhs)

    qrhs = aqt_tensor.QTensor(
        qvalue=None,
        scale=[xrhs_scale],
        scale_t=None,
        bias=[],
        dequant_dtype=xrhs_scale.dtype,
        tiling_state=xrhs,
    )
    qrhs = qrhs.quant(rhs)

    self.assertEqual(qlhs.shape, lhs.shape)
    self.assertEqual(qrhs.shape, rhs.shape)
    self.assertEqual(qlhs.ndim, lhs.ndim)
    self.assertEqual(qrhs.ndim, rhs.ndim)

    self.assertEqual(qlhs.qvalue.shape, tuple(xlhs.tiled_shape))
    self.assertEqual(qrhs.qvalue.shape, tuple(xrhs.tiled_shape))

    self.assertEqual(qlhs.dequant().shape, lhs.shape)
    self.assertEqual(qrhs.dequant().shape, rhs.shape)

    # Check assertion raised when wrong tiling_state is given.
    qlhs.tiling_state.tiled_shape = qrhs.qvalue.shape
    with self.assertRaises(AssertionError):
      qlhs.dequant()
    with self.assertRaises(AssertionError):
      qlhs.shape  # pylint: disable=pointless-statement

    qrhs.tiling_state.tiled_shape = qlhs.qvalue.shape
    with self.assertRaises(AssertionError):
      qrhs.dequant()
    with self.assertRaises(AssertionError):
      qrhs.shape  # pylint: disable=pointless-statement


if __name__ == "__main__":
  absltest.main()
