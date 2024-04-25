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

"""Test for AQT pallas."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax.v2 import aqt_pallas
from aqt.jax.v2 import aqt_tensor
import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np


class AqtPallasTest(parameterized.TestCase):

  @parameterized.parameters(
      ((512, 512), (0,), (128, 128), (1, 512), (1, 128)),
      ((512, 512), (1,), (128, 128), (512, 1), (128, 1)),
      (
          (512, 512, 1024),
          (1, 2),
          (128, 128, 128),
          (512, 1, 1),
          (128, 1, 1),
      ),
  )
  def test_quant_correctly(
      self,
      tensor_shape,
      calibration_axes,
      block_shape,
      expected_scale_shape,
      expected_scale_block_shape,
  ):
    """Test whether QTenor can be used as an argument in pallas."""
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, tensor_shape)
    block_spec = pl.BlockSpec(lambda *args: args, block_shape)

    qx, qx_blockspec = aqt_pallas.quant(
        x, n_bits=8, calibration_axes=calibration_axes, block_spec=block_spec
    )
    self.assertEqual(qx.qvalue.shape, x.shape)
    self.assertEqual(qx.scale[0].shape, expected_scale_shape)
    self.assertIsNone(qx.scale_permute_axis)
    self.assertIsNone(qx.scale_t)
    self.assertEqual(
        qx_blockspec.scale[0].block_shape, expected_scale_block_shape
    )

  @parameterized.parameters(
      ((512, 512), (0,), (128, 128), (4, 1, 512), (1, 1, 128), [0, -1, 1]),
      ((512, 512), (1,), (128, 128), (4, 1, 512), (1, 1, 128), [1, -1, 0]),
      (
          (512, 512, 1024),
          (1, 2),
          (128, 128, 128),
          (4, 8, 1, 512),
          (1, 1, 1, 128),
          [1, 2, -1, 0],
      ),
  )
  def test_quant_blockwisely_correctness(
      self,
      tensor_shape,
      calibration_axes,
      block_shape,
      expected_scale_shape,
      expected_scale_block_shape,
      expected_scale_permute_axis,
  ):
    """Test whether QTenor can be used as an argument in pallas."""
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, tensor_shape)
    block_spec = pl.BlockSpec(lambda *args: args, block_shape)

    qx, qx_blockspec = aqt_pallas.quant_blockwisely(
        x, n_bits=8, calibration_axes=calibration_axes, block_spec=block_spec
    )
    self.assertEqual(qx.qvalue.shape, x.shape)
    self.assertEqual(qx.scale[0].shape, expected_scale_shape)
    self.assertEqual(qx.scale_permute_axis[0], expected_scale_permute_axis)
    self.assertIsNone(qx.scale_t)
    self.assertEqual(
        qx_blockspec.scale[0].block_shape, expected_scale_block_shape
    )

  @parameterized.product(
      quant_spec=[
          (
              (1024, 1024),
              (1,),
              (256, 256),
          ),
          (
              (1024, 1024),
              (0,),
              (256, 256),
          ),
          (
              (10, 512, 1024),
              (1,),
              (1, 256, 256),
          ),
          (
              (10, 512, 1024),
              (2,),
              (1, 256, 256),
          ),
      ],
      block_wise=[True, False],
  )
  def test_quant_dequant(self, quant_spec, block_wise=True):
    """Test whether QTenor can be used as an argument in pallas."""
    tensor_shape, calibration_axes, block_shape = quant_spec

    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, tensor_shape, minval=-3, maxval=3)
    block_spec = pl.BlockSpec(lambda *args: args, block_shape)

    quant_func = (
        aqt_pallas.quant if block_wise else aqt_pallas.quant_blockwisely
    )

    @functools.partial(jax.jit, static_argnames=["block_spec"])
    def quant_dequant(x, block_spec):
      qx, qx_blockspec = quant_func(
          x,
          n_bits=8,
          calibration_axes=calibration_axes,
          block_spec=block_spec,
      )
      grid = [
          ndim // blk_ndim for ndim, blk_ndim in zip(tensor_shape, block_shape)
      ]

      def dequant_kernel(qx: aqt_tensor.QTensor, out_ref):
        qx = aqt_pallas.materialize_qtensor(qx)
        out_ref[...] = qx.dequant()

      dequant_out = pl.pallas_call(
          dequant_kernel,
          grid=tuple(grid),
          in_specs=[qx_blockspec],
          out_specs=block_spec,
          out_shape=jax.ShapeDtypeStruct(shape=tensor_shape, dtype=jnp.float32),
          interpret=False,
      )(qx)
      return dequant_out

    np.testing.assert_array_almost_equal(
        quant_dequant(x, block_spec), x, decimal=1
    )


if __name__ == "__main__":
  absltest.main()
