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

"""Tests for sparsity."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax_legacy.jax.sparsity import sparsity_core as sparsity
from aqt.jax_legacy.jax.sparsity import sparsity_hparams
from jax import numpy as jnp
import numpy as np


dataclass = dataclasses.dataclass


class PruningParamsTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(sparse_type='structured_nm', prune_rate=0.1),
      dict(sparse_type='unstructured', prune_rate=(1, 4)),
  )
  def test_invalid_params(self, sparse_type, prune_rate):
    with self.assertRaisesRegex(
        AssertionError, 'Prune rate must be either None'
    ):
      weight_params = sparsity_hparams.WeightSparsityParams(
          prune_rate=prune_rate
      )
      sparsity_hparams.SparsityHParams(
          sparsity_type=sparse_type, weight_params=weight_params
      )

  @parameterized.parameters(
      dict(prune_rate=(1, 4), mask_decay_weight=-0.1),
      dict(prune_rate=0.2, mask_decay_weight=-0.1),
  )
  def test_invalid_mask_decay_weight(self, prune_rate, mask_decay_weight):
    with self.assertRaisesRegex(
        AssertionError, '.* `mask_decay_weight` must be positive.'
    ):
      sparsity_hparams.WeightSparsityParams(
          prune_rate=prune_rate, mask_decay_weight=mask_decay_weight
      )

  @parameterized.parameters(
      dict(
          prune_rate=(1, 4),
          sparse_ste=True,
          mask_decay_weight=0.1,
      ),
      dict(
          prune_rate=0.2,
          sparse_ste=True,
          mask_decay_weight=0.1,
      ),
  )
  def test_invalid_sparse_ste_with_non_zero_mask_decay_weight(
      self, prune_rate, sparse_ste, mask_decay_weight
  ):
    with self.assertRaisesRegex(
        ValueError, 'SR-STE only works with non-decaying mask.'
    ):
      sparsity_hparams.WeightSparsityParams(
          prune_rate=prune_rate,
          sparse_ste=sparse_ste,
          mask_decay_weight=mask_decay_weight,
      )

  @parameterized.parameters(
      dict(
          prune_rate=(1, 4),
          sparse_ste=True,
          structure_decay=True,
      ),
      dict(
          prune_rate=0.2,
          sparse_ste=True,
          structure_decay=True,
      ),
  )
  def test_invalid_sparse_ste_with_structure_decay(
      self, prune_rate, sparse_ste, structure_decay
  ):
    with self.assertRaisesRegex(
        ValueError, 'SR-STE only works with non-decaying sparse structure.'
    ):
      sparsity_hparams.WeightSparsityParams(
          prune_rate=prune_rate,
          sparse_ste=sparse_ste,
          structure_decay=structure_decay,
      )

  @parameterized.parameters(
      dict(sparse_type='unstructured', prune_rate=0.2, sparse_ste=True)
  )
  def test_invalid_sparse_ste_with_unstructured_sparsity(
      self, sparse_type, prune_rate, sparse_ste
  ):
    with self.assertRaisesRegex(
        ValueError, 'SR-STE only works with structured sparsity.'
    ):
      weight_params = sparsity_hparams.WeightSparsityParams(
          prune_rate=prune_rate, sparse_ste=sparse_ste
      )
      sparsity_hparams.SparsityHParams(
          sparsity_type=sparse_type, weight_params=weight_params
      )

  def test_invalid_prune_rate(self):
    error_msg = 'N must be lower than M'
    inputs = jnp.arange(12)
    with self.assertRaisesRegex(AssertionError, error_msg):
      sparsity.get_sparsity_mask(inputs, n_sparsity=4, m_sparsity=1)


class ChannelwisePruningTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          inputs=np.reshape(np.arange(1, 9), (2, 4)),
          testcase_name='row_wise_pruning',
          channel_dim=-2,
          has_stacked_layers=False,
          exp_mask=[[0], [1]],
          exp_pruned_inputs=[[5, 6, 7, 8]],
      ),
      dict(
          inputs=np.reshape(np.arange(1, 9), (2, 4)),
          testcase_name='column_wise_pruning',
          channel_dim=-1,
          has_stacked_layers=False,
          exp_mask=[[0, 0, 1, 1]],
          exp_pruned_inputs=[[3, 4], [7, 8]],
      ),
      dict(
          inputs=np.reshape(
              np.concatenate((np.arange(1, 13), np.arange(12, 0, -1))),
              (2, 3, 4),
          ),
          testcase_name='per_layer_channel_pruning',
          channel_dim=-1,
          has_stacked_layers=True,
          exp_mask=[
              [[0, 0, 1, 1]],
              [[1, 1, 0, 0]],
          ],
          exp_pruned_inputs=[
              [[3, 4], [7, 8], [11, 12]],
              [[12, 11], [8, 7], [4, 3]],
          ],
      ),
  )
  def test_pruning(
      self, inputs, channel_dim, has_stacked_layers, exp_mask, exp_pruned_inputs
  ):
    inputs = jnp.array(inputs)
    mask = sparsity.get_sparsity_mask_channelwise(
        inputs, 0.5, channel_dim, has_stacked_layers=has_stacked_layers
    )
    print(mask)
    np.testing.assert_array_equal(mask, exp_mask)

    pruned_inputs = sparsity.prune_channels(inputs, channel_dim, mask)
    np.testing.assert_array_equal(pruned_inputs, exp_pruned_inputs)


class ChannelwiseMaskingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='row_wise_pruning',
          channel_dim=-2,
          exp_mask=[
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          ],
      ),
      dict(
          testcase_name='column_wise_pruning',
          channel_dim=-1,
          exp_mask=[
              [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
          ],
      ),
  )
  def test_column_row_pruning(self, channel_dim, exp_mask):
    inputs = jnp.reshape(jnp.arange(1, 73), (6, 12))
    mask = sparsity.get_sparsity_mask_channelwise(inputs, 0.5, channel_dim)
    np.testing.assert_array_equal(mask, exp_mask)

  @parameterized.named_parameters(
      dict(
          testcase_name='row_wise_pruning',
          channel_dim=-2,
          exp_mask=[
              [1, 1, 1, 1],
              [1, 1, 1, 1],
              [1, 1, 1, 1],
              [0, 0, 0, 0],
          ],
      ),
      dict(
          testcase_name='column_wise_pruning',
          channel_dim=-1,
          exp_mask=[
              [1, 1, 1, 0],
              [1, 1, 1, 0],
              [1, 1, 1, 0],
              [1, 1, 1, 0],
          ],
      ),
  )
  def test_column_row_pruning_same_score(self, channel_dim, exp_mask):
    inputs = jnp.ones((4, 4))
    mask = sparsity.get_sparsity_mask_channelwise(
        inputs, 0.25, channel_dim
    ).astype(jnp.int32)
    np.testing.assert_array_equal(mask, exp_mask)

  @parameterized.named_parameters(
      dict(
          testcase_name='row_wise_pruning',
          channel_dim=-2,
          exp_mask=[
              [
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              ],
              [
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              ],
              [
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              ],
          ],
      ),
  )
  def test_3d_pruning(self, channel_dim, exp_mask):
    inputs = jnp.reshape(jnp.arange(1, 73), (3, 2, 12))
    mask = sparsity.get_sparsity_mask_channelwise(inputs, 0.5, channel_dim)
    np.testing.assert_array_equal(mask, exp_mask)


class PruningFunctionalityTest(parameterized.TestCase):

  def test_prune_inputs_n_m(self):
    inputs = jnp.array(np.random.rand(10, 2, 4))
    prune_rate = (1, 4)

    out = sparsity.prune_inputs_n_m(
        inputs, n=prune_rate[0], m=prune_rate[1], order='R'
    )
    self.assertEqual(out.shape[0], inputs.shape[0])
    self.assertEqual(out.shape[1], inputs.shape[1])
    self.assertEqual(out.shape[2], inputs.shape[2])

    # Only 20 non-zero elements must exist after pruning.
    num_non_zero_elems = 0.25 * inputs.size
    self.assertEqual(out[out != 0].shape[0], num_non_zero_elems)
    self.assertEqual(
        list(np.argmax(inputs, axis=2).flatten()),
        list(np.argmax(out != 0, axis=2).flatten()),
    )

  def test_n_m_pruning_mask(self):
    inputs = jnp.array(np.random.rand(10, 2, 4))
    prune_rate = (1, 4)
    mask = sparsity.get_sparsity_mask(
        inputs, n_sparsity=prune_rate[0], m_sparsity=prune_rate[1], order='R'
    )
    self.assertEqual(
        list(np.argmax(inputs, axis=2).flatten()),
        list(np.argmax(mask == 1, axis=2).flatten()),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='2d_row_wise_pruning',
          order='R',
          inputs=np.arange(1, 73).reshape(6, 12),
          exp_output=[
              [0, 2, 3, 0, 5, 6, 0, 8, 9, 0, 11, 12],
              [0, 14, 15, 0, 17, 18, 0, 20, 21, 0, 23, 24],
              [0, 26, 27, 0, 29, 30, 0, 32, 33, 0, 35, 36],
              [0, 38, 39, 0, 41, 42, 0, 44, 45, 0, 47, 48],
              [0, 50, 51, 0, 53, 54, 0, 56, 57, 0, 59, 60],
              [0, 62, 63, 0, 65, 66, 0, 68, 69, 0, 71, 72],
          ],
          n_sparsity=2,
          m_sparsity=3,
      ),
      dict(
          testcase_name='3d_row_wise_pruning',
          order='R',
          inputs=np.arange(1, 73).reshape(2, 6, 6),
          exp_output=[
              [
                  [0, 2, 3, 0, 5, 6],
                  [0, 8, 9, 0, 11, 12],
                  [0, 14, 15, 0, 17, 18],
                  [0, 20, 21, 0, 23, 24],
                  [0, 26, 27, 0, 29, 30],
                  [0, 32, 33, 0, 35, 36],
              ],
              [
                  [0, 38, 39, 0, 41, 42],
                  [0, 44, 45, 0, 47, 48],
                  [0, 50, 51, 0, 53, 54],
                  [0, 56, 57, 0, 59, 60],
                  [0, 62, 63, 0, 65, 66],
                  [0, 68, 69, 0, 71, 72],
              ],
          ],
          n_sparsity=2,
          m_sparsity=3,
      ),
      dict(
          testcase_name='2d_column_wise_pruning',
          order='C',
          inputs=np.arange(1, 73).reshape(6, 12),
          exp_output=[
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
              [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
              [61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72],
          ],
          n_sparsity=2,
          m_sparsity=3,
      ),
      dict(
          testcase_name='3d_column_wise_pruning',
          order='C',
          inputs=np.arange(1, 65).reshape(4, 4, 4),
          exp_output=[
              [
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16],
              ],
              [
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [25, 26, 27, 28],
                  [29, 30, 31, 32],
              ],
              [
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [41, 42, 43, 44],
                  [45, 46, 47, 48],
              ],
              [
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [57, 58, 59, 60],
                  [61, 62, 63, 64],
              ],
          ],
          n_sparsity=2,
          m_sparsity=4,
      ),
      dict(
          testcase_name='3d_column_wise_pruning2',
          order='C',
          inputs=np.arange(1, 33).reshape(2, 4, 4),
          exp_output=[
              [
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16],
              ],
              [
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [25, 26, 27, 28],
                  [29, 30, 31, 32],
              ],
          ],
          n_sparsity=2,
          m_sparsity=4,
      ),
  )
  def test_pruning(self, order, inputs, exp_output, n_sparsity, m_sparsity):
    inputs = jnp.array(inputs)
    output = sparsity.prune_inputs_n_m(
        inputs, n=n_sparsity, m=m_sparsity, order=order
    )
    np.testing.assert_array_equal(output, exp_output)


class BlockPruningFunctionalityTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='block_size_1', block_size=1),
      dict(testcase_name='block_size_2', block_size=2),
      dict(testcase_name='block_size_4', block_size=4),
  )
  def test_prune_inputs_n_m(self, block_size):
    inputs = jnp.array(np.random.rand(10, 2, 4))
    prune_rate = (1, 2)

    block_mask = sparsity.get_sparsity_mask(
        inputs,
        n_sparsity=prune_rate[0],
        m_sparsity=prune_rate[1],
        order='R',
        block_size=block_size,
    )
    out = sparsity.apply_sparsity(inputs, block_mask)
    self.assertEqual(out.shape[0], inputs.shape[0])
    self.assertEqual(out.shape[1], inputs.shape[1])
    self.assertEqual(out.shape[2], inputs.shape[2])

    # Only 40 non-zero elements must exist after pruning.
    num_non_zero_elems = 0.5 * inputs.size
    self.assertEqual(out[out != 0].shape[0], num_non_zero_elems)

  @parameterized.named_parameters(
      dict(
          testcase_name='2d_row_wise_pruning',
          order='R',
          inputs=np.arange(1, 73).reshape(6, 12),
          exp_output=[
              [0, 0, 3, 4, 5, 6, 0, 0, 9, 10, 11, 12],
              [0, 0, 15, 16, 17, 18, 0, 0, 21, 22, 23, 24],
              [0, 0, 27, 28, 29, 30, 0, 0, 33, 34, 35, 36],
              [0, 0, 39, 40, 41, 42, 0, 0, 45, 46, 47, 48],
              [0, 0, 51, 52, 53, 54, 0, 0, 57, 58, 59, 60],
              [0, 0, 63, 64, 65, 66, 0, 0, 69, 70, 71, 72],
          ],
          n_sparsity=2,
          m_sparsity=3,
          block_size=2,
      ),
      dict(
          testcase_name='2d_row_wise_pruning_2',
          order='R',
          inputs=np.arange(1, 73).reshape(6, 12),
          exp_output=[
              [0, 0, 0, 0, 5, 6, 7, 8, 9, 10, 11, 12],
              [0, 0, 0, 0, 17, 18, 19, 20, 21, 22, 23, 24],
              [0, 0, 0, 0, 29, 30, 31, 32, 33, 34, 35, 36],
              [0, 0, 0, 0, 41, 42, 43, 44, 45, 46, 47, 48],
              [0, 0, 0, 0, 53, 54, 55, 56, 57, 58, 59, 60],
              [0, 0, 0, 0, 65, 66, 67, 68, 69, 70, 71, 72],
          ],
          n_sparsity=2,
          m_sparsity=3,
          block_size=4,
      ),
      dict(
          testcase_name='3d_row_wise_pruning',
          order='R',
          inputs=np.arange(1, 73).reshape(2, 6, 6),
          exp_output=[
              [
                  [0, 0, 3, 4, 5, 6],
                  [0, 0, 9, 10, 11, 12],
                  [0, 0, 15, 16, 17, 18],
                  [0, 0, 21, 22, 23, 24],
                  [0, 0, 27, 28, 29, 30],
                  [0, 0, 33, 34, 35, 36],
              ],
              [
                  [0, 0, 39, 40, 41, 42],
                  [0, 0, 45, 46, 47, 48],
                  [0, 0, 51, 52, 53, 54],
                  [0, 0, 57, 58, 59, 60],
                  [0, 0, 63, 64, 65, 66],
                  [0, 0, 69, 70, 71, 72],
              ],
          ],
          n_sparsity=2,
          m_sparsity=3,
          block_size=2,
      ),
      dict(
          testcase_name='3d_row_wise_pruning_2',
          order='R',
          inputs=np.arange(1, 73).reshape(1, 6, 12),
          exp_output=[
              [
                  [0, 0, 0, 0, 5, 6, 7, 8, 9, 10, 11, 12],
                  [0, 0, 0, 0, 17, 18, 19, 20, 21, 22, 23, 24],
                  [0, 0, 0, 0, 29, 30, 31, 32, 33, 34, 35, 36],
                  [0, 0, 0, 0, 41, 42, 43, 44, 45, 46, 47, 48],
                  [0, 0, 0, 0, 53, 54, 55, 56, 57, 58, 59, 60],
                  [0, 0, 0, 0, 65, 66, 67, 68, 69, 70, 71, 72],
              ],
          ],
          n_sparsity=2,
          m_sparsity=3,
          block_size=4,
      ),
      dict(
          testcase_name='2d_column_wise_pruning',
          order='C',
          inputs=np.arange(1, 73).reshape(6, 12),
          exp_output=[
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
              [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48],
              [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
              [61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72],
          ],
          n_sparsity=2,
          m_sparsity=3,
          block_size=2,
      ),
      dict(
          testcase_name='2d_column_wise_pruning_2',
          order='C',
          inputs=np.arange(1, 73).reshape(12, 6),
          exp_output=[
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [25, 26, 27, 28, 29, 30],
              [31, 32, 33, 34, 35, 36],
              [37, 38, 39, 40, 41, 42],
              [43, 44, 45, 46, 47, 48],
              [49, 50, 51, 52, 53, 54],
              [55, 56, 57, 58, 59, 60],
              [61, 62, 63, 64, 65, 66],
              [67, 68, 69, 70, 71, 72],
          ],
          n_sparsity=2,
          m_sparsity=3,
          block_size=4,
      ),
      dict(
          testcase_name='3d_column_wise_pruning',
          order='C',
          inputs=np.arange(1, 65).reshape(2, 8, 4),
          exp_output=[
              [
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [17, 18, 19, 20],
                  [21, 22, 23, 24],
                  [25, 26, 27, 28],
                  [29, 30, 31, 32],
              ],
              [
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [49, 50, 51, 52],
                  [53, 54, 55, 56],
                  [57, 58, 59, 60],
                  [61, 62, 63, 64],
              ],
          ],
          n_sparsity=2,
          m_sparsity=4,
          block_size=2,
      ),
      dict(
          testcase_name='3d_column_wise_pruning_2',
          order='C',
          inputs=np.arange(1, 65).reshape(1, 16, 4),
          exp_output=[
              [
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [33, 34, 35, 36],
                  [37, 38, 39, 40],
                  [41, 42, 43, 44],
                  [45, 46, 47, 48],
                  [49, 50, 51, 52],
                  [53, 54, 55, 56],
                  [57, 58, 59, 60],
                  [61, 62, 63, 64],
              ],
          ],
          n_sparsity=2,
          m_sparsity=4,
          block_size=4,
      ),
  )
  def test_block_pruning(
      self, order, inputs, exp_output, n_sparsity, m_sparsity, block_size
  ):
    inputs = jnp.array(inputs)
    block_mask = sparsity.get_sparsity_mask(
        inputs, n_sparsity, m_sparsity, order=order, block_size=block_size
    )
    output = sparsity.apply_sparsity(inputs, block_mask)
    np.testing.assert_array_equal(output, exp_output)


class PruningScoreTest(parameterized.TestCase):

  def test_score_activation_weighted(self):
    weight = jnp.array([[1.0, 2.0, 3.0, 4.0], [-4.0, -3.0, -2.0, -1.0]])
    activation = jnp.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    expected_score = jnp.array([[4.0, 8.0, 12.0, 16.0], [0.0, 0.0, 0.0, 0.0]])
    score = sparsity.compute_score(
        weight,
        inputs=activation,
        score_func=sparsity_hparams.SparsityScore.ACTIVATION_WEIGHTED,
    )
    self.assertTrue((score == expected_score).all())


if __name__ == '__main__':
  absltest.main()
