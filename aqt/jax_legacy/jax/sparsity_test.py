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
import typing

from absl.testing import absltest
from absl.testing import parameterized

from aqt.jax_legacy.jax.flax import struct as flax_struct
import aqt.jax_legacy.jax.sparse_context as SparseContext
from aqt.jax_legacy.jax.sparsity import SparseHParams
from aqt.jax_legacy.jax.sparsity import Sparsity

from flax import linen as nn

import jax
from jax import numpy as jnp
from jax import random

import numpy as np


dataclass = (
    flax_struct.dataclass if not typing.TYPE_CHECKING else dataclasses.dataclass
)


class SparsityTest(parameterized.TestCase):

  def init_model(
      self,
      update_mask,
      apply_mask,
      sparsity_type,
      structure_decay=False,
      num_update_sparsity=0,
      mask_decay_weight=0.0,
      prune_rate=(2, 4),
      sparse_ste=False,
  ):
    rng = random.PRNGKey(0)
    self.inputs = jnp.array([[3, 4, 6], [1, 2, 1]])
    if sparsity_type == 0:
      sparsity_hparams = SparseHParams(
          type='STRUCTURED_NM',
          prune_rate=prune_rate,
          structure_decay=structure_decay,
          mask_decay_weight=mask_decay_weight,
          sparse_ste=sparse_ste,
      )
    elif sparsity_type == 1:
      sparsity_hparams = SparseHParams(
          type='UNSTRUCTURED',
          prune_rate=prune_rate,
          structure_decay=structure_decay,
          mask_decay_weight=mask_decay_weight,
          sparse_ste=sparse_ste,
      )
    elif sparsity_type == 2:
      sparsity_hparams = SparseHParams(
          type='STRUCTURED_NMC',
          prune_rate=prune_rate,
          structure_decay=structure_decay,
          mask_decay_weight=mask_decay_weight,
          sparse_ste=sparse_ste,
      )
    else:
      assert False

    sparsity_module = Sparsity(sparsity_hparams=sparsity_hparams)
    init_mask = sparsity_module.init(
        rng,
        self.inputs,
        update_mask=update_mask,
        apply_mask=apply_mask,
        num_update_sparsity=num_update_sparsity,
    )
    return sparsity_module, init_mask

  @parameterized.named_parameters(
      dict(
          testcase_name='structured_nm',
          sparsity_type=0,
          prune_rate=(2, 4),
      ),
      dict(
          testcase_name='unstructured',
          sparsity_type=1,
          prune_rate=0.1,
      ),
      dict(
          testcase_name='structured_nmc',
          sparsity_type=2,
          prune_rate=(2, 2, 2),
      ),
  )
  def test_init(self, sparsity_type, prune_rate):
    update_mask = SparseContext.DynamicContext(update_mask=False)
    _, init_state = self.init_model(
        apply_mask=False,
        prune_rate=prune_rate,
        update_mask=update_mask,
        sparsity_type=sparsity_type,
    )
    init_state_mask = init_state['sparsity']['mask']
    np.testing.assert_array_equal(init_state_mask, [[0, 0, 0], [0, 0, 0]])

  def test_structured_nm_with_no_pruning(self):
    update_mask = SparseContext.DynamicContext(update_mask=False)
    model, init_state = self.init_model(
        update_mask=update_mask,
        apply_mask=False,
        sparsity_type=0,
        structure_decay=False,
        prune_rate=(4, 4),
    )
    np.testing.assert_array_equal(
        init_state['sparsity']['mask'], jnp.array([[0, 0, 0], [0, 0, 0]])
    )
    # We need inputs that are divisible by four.
    inputs = jnp.array([[3, 4, 6, 8], [1, 2, 1, 4]])
    update_mask = SparseContext.DynamicContext(update_mask=True)
    model_out, state_0 = model.apply(
        init_state,
        inputs,
        update_mask=update_mask,
        apply_mask=True,
        mutable='sparsity',
    )
    np.testing.assert_array_equal(
        model_out, jnp.array([[3, 4, 6, 8], [1, 2, 1, 4]])
    )
    np.testing.assert_array_equal(
        state_0['sparsity']['mask'], jnp.array([[1, 1, 1, 1], [1, 1, 1, 1]])
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='structured_nmc_2_2_2',
          inputs=[[3, 4, 6, 1], [1, 2, 1, 4]],
          outputs=[[3, 4, 6, 0], [0, 0, 0, 4]],
          mask=[[1, 1, 1, 0], [0, 0, 0, 1]],
          prunt_rate=(2, 2, 2),
      ),
      dict(
          testcase_name='structured_nmc_2_2_4',
          inputs=[[3, 4, 6, 1], [1, 2, 1, 5], [-1, 2, 6, -4], [0, 2, -7, -2]],
          outputs=[[0, 0, 6, 0], [0, 0, 0, 5], [0, 0, 6, 0], [0, 0, -7, 0]],
          mask=[[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0]],
          prunt_rate=(2, 2, 4),
      ),
      dict(
          testcase_name='structured_nmc_1_4_2',
          inputs=[[3, 4, 6, 1], [1, 2, 1, 5], [-1, 2, 6, -4], [0, 2, -7, -2]],
          outputs=[[0, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, -7, 0]],
          mask=[[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]],
          prunt_rate=(1, 4, 2),
      ),
  )
  def test_structured_nmc(self, inputs, outputs, mask, prunt_rate):
    update_mask = SparseContext.DynamicContext(update_mask=False)
    model, init_state = self.init_model(
        update_mask=update_mask,
        apply_mask=False,
        sparsity_type=2,
        structure_decay=False,
        prune_rate=prunt_rate,
    )
    # We need inputs that are divisible by four.
    jin = jnp.array(inputs)
    update_mask = SparseContext.DynamicContext(update_mask=True)
    model_out, state_0 = model.apply(
        init_state,
        jin,
        update_mask=update_mask,
        apply_mask=True,
        mutable='sparsity',
    )
    np.testing.assert_array_equal(model_out, jnp.array(outputs))
    np.testing.assert_array_equal(state_0['sparsity']['mask'], jnp.array(mask))

  @parameterized.named_parameters(
      dict(
          testcase_name='initial_structure_decay',
          num_update_sparsity=0,
          out=[[3, 4, 6, 8], [1, 2, 1, 4]],
          mask=[[1, 1, 1, 1], [1, 1, 1, 1]],
      ),
      dict(
          testcase_name='first_iteration_structure_decay',
          num_update_sparsity=1,
          out=[[0, 4, 6, 8], [0, 2, 1, 4]],
          mask=[[0, 1, 1, 1], [0, 1, 1, 1]],
      ),
      dict(
          testcase_name='second_iteration_structure_decay',
          num_update_sparsity=2,
          out=[[0, 0, 0, 8], [0, 0, 0, 4]],
          mask=[[0, 0, 0, 1], [0, 0, 0, 1]],
      ),
      dict(
          testcase_name='third_iteration_structure_decay',
          num_update_sparsity=3,
          out=[[0, 0, 0, 8], [0, 0, 0, 4]],
          mask=[[0, 0, 0, 1], [0, 0, 0, 1]],
      ),
  )
  def test_structure_decay(self, num_update_sparsity, out, mask):
    update_mask = SparseContext.DynamicContext(update_mask=False)
    model, init_state = self.init_model(
        update_mask=update_mask,
        apply_mask=False,
        sparsity_type=0,
        structure_decay=True,
        prune_rate=(4, 4))
    # We need inputs that are divisible by four.
    inputs = jnp.array([[3, 4, 6, 8], [1, 2, 1, 4]])
    update_mask = SparseContext.DynamicContext(update_mask=True)
    model_out, state_0 = model.apply(
        init_state,
        inputs,
        update_mask=update_mask,
        apply_mask=True,
        num_update_sparsity=num_update_sparsity,
        mutable='sparsity')
    np.testing.assert_array_equal(model_out, out)
    np.testing.assert_array_equal(state_0['sparsity']['mask'], mask)

  @parameterized.named_parameters(
      dict(
          testcase_name='initial_mask_decay',
          num_update_sparsity=0,
          out=[[3, 4, 6, 8], [1, 2, 1, 4]],
          mask=[[0, 0, 1, 1], [0, 1, 0, 1]]),
      dict(
          testcase_name='first_iteration_mask_decay',
          num_update_sparsity=1,
          out=[[0.9 * 3, 0.9 * 4, 6, 8], [0.9 * 1, 2, 0.9 * 1, 4]],
          mask=[[0, 0, 1, 1], [0, 1, 0, 1]]),
      dict(
          testcase_name='second_iteration_mask_decay',
          num_update_sparsity=2,
          out=[[0.8 * 3, 0.8 * 4, 6, 8], [0.8 * 1, 2, 0.8 * 1, 4]],
          mask=[[0, 0, 1, 1], [0, 1, 0, 1]]),
      dict(
          testcase_name='third_iteration_mask_decay',
          num_update_sparsity=3,
          out=[[0.7 * 3, 0.7 * 4, 6, 8], [0.7 * 1, 2, 0.7 * 1, 4]],
          mask=[[0, 0, 1, 1], [0, 1, 0, 1]]),
      dict(
          testcase_name='ninth_iteration_mask_decay',
          num_update_sparsity=9,
          out=[[0.1 * 3, 0.1 * 4, 6, 8], [0.1 * 1, 2, 0.1 * 1, 4]],
          mask=[[0, 0, 1, 1], [0, 1, 0, 1]]),
      dict(
          testcase_name='tenth_iteration_mask_decay',
          num_update_sparsity=10,
          out=[[0.0 * 3, 0.0 * 4, 6, 8], [0.0 * 1, 2, 0.0 * 1, 4]],
          mask=[[0, 0, 1, 1], [0, 1, 0, 1]]),
  )
  def test_mask_decay(self, num_update_sparsity, out, mask):
    update_mask = SparseContext.DynamicContext(update_mask=False)
    model, init_state = self.init_model(
        update_mask=update_mask,
        apply_mask=False,
        sparsity_type=0,
        structure_decay=False,
        mask_decay_weight=0.1,
        prune_rate=(2, 4))
    # We need inputs that are divisible by four.
    inputs = jnp.array([[3., 4., 6., 8.], [1., 2., 1., 4.]])
    update_mask = SparseContext.DynamicContext(update_mask=True)
    model_out, state_0 = model.apply(
        init_state,
        inputs,
        update_mask=update_mask,
        apply_mask=True,
        num_update_sparsity=num_update_sparsity,
        mutable='sparsity')
    np.testing.assert_allclose(model_out, out)
    np.testing.assert_array_equal(state_0['sparsity']['mask'], mask)

  @parameterized.named_parameters(
      dict(
          testcase_name='update_mask_apply_mask',
          update_mask=SparseContext.DynamicContext(update_mask=True),
          apply_mask=True),
      dict(
          testcase_name='no_update_mask_apply_mask',
          update_mask=SparseContext.DynamicContext(update_mask=False),
          apply_mask=True),
      dict(
          testcase_name='update_mask_no_apply_mask',
          update_mask=SparseContext.DynamicContext(update_mask=True),
          apply_mask=False),
      dict(
          testcase_name='no_update_mask_no_apply_mask',
          update_mask=SparseContext.DynamicContext(update_mask=False),
          apply_mask=False),
  )
  def test_sr_ste_fwd_pass(self, update_mask, apply_mask):
    rng = random.PRNGKey(0)
    sparsity_hparams = SparseHParams(
        type='STRUCTURED_NM',
        prune_rate=(2, 4),
        sparse_ste=True,
        structure_decay=False,
        mask_decay_weight=0.0,
    )
    inputs = jnp.array([[3., 4., 6., 8.], [1., 2., 1., 4.]])
    sparsity_module = Sparsity(sparsity_hparams=sparsity_hparams)
    init_state = sparsity_module.init(
        rng,
        inputs,
        update_mask=SparseContext.DynamicContext(update_mask=False),
        apply_mask=True,
        num_update_sparsity=0.0)
    # We need inputs that are divisible by four.
    inputs = jnp.array([[3., 4., 6., 8.], [1., 2., 1., 4.]])
    out, state_0 = sparsity_module.apply(
        init_state,
        inputs,
        update_mask=update_mask,
        apply_mask=apply_mask,
        mutable='sparsity',
    )
    state_0_mask = state_0['sparsity']['mask']
    if not apply_mask:
      np.testing.assert_array_equal(out, inputs)
    elif apply_mask and not update_mask.update_mask:
      np.testing.assert_array_equal(
          out, jnp.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
      )
    elif apply_mask and update_mask.update_mask:
      np.testing.assert_array_equal(out, [[0, 0, 6, 8], [0, 2, 0, 4]])

    if update_mask.update_mask:
      np.testing.assert_array_equal(
          state_0_mask,
          [[0, 0, 1, 1], [0, 1, 0, 1]])
    else:
      np.testing.assert_array_equal(
          state_0_mask, [[0, 0, 0, 0], [0, 0, 0, 0]])

    inputs2 = jnp.array([[2., 3., 1., 1.], [2., -6., 1., 5.]])
    out, state = sparsity_module.apply(
        state_0,
        inputs2,
        update_mask=update_mask,
        apply_mask=apply_mask,
        mutable='sparsity',
    )
    state_mask = state['sparsity']['mask']
    if not apply_mask:
      np.testing.assert_array_equal(out, inputs2)
    elif apply_mask and not update_mask.update_mask:
      np.testing.assert_array_equal(
          out, jnp.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
      )
    elif apply_mask and update_mask.update_mask:
      np.testing.assert_array_equal(out, [[2, 3, 0, 0], [0, -6, 0, 5]])
    if update_mask.update_mask:
      np.testing.assert_array_equal(state_mask, [[1, 1, 0, 0], [0, 1, 0, 1]])
    else:
      np.testing.assert_array_equal(state_mask, [[0, 0, 0, 0], [0, 0, 0, 0]])

  def test_sr_ste_bwd_pass(self):
    class SingleLayer(nn.Module):

      @dataclass
      class HParams:
        sparsity: SparseHParams

      hparams: HParams

      @nn.compact
      def __call__(
          self,
          inputs: jnp.ndarray,
          update_mask: SparseContext.DynamicContext,
          apply_mask: bool,
      ) -> jnp.ndarray:
        kernel = self.param('kernel', nn.initializers.ones, (1, 4))
        kernel = Sparsity(
            sparsity_hparams=self.hparams.sparsity, name='weight_sparsity'
        )(
            kernel,
            update_mask=update_mask,
            apply_mask=apply_mask,
            num_update_sparsity=0,
        )
        return jnp.multiply(inputs, kernel)

    rng = random.PRNGKey(0)
    layer_kwargs = {}
    layer_kwargs['hparams'] = SingleLayer.HParams(
        sparsity=SparseHParams(
            type='STRUCTURED_NM',
            prune_rate=(2, 4),
            sparse_ste=True,
            structure_decay=False,
            mask_decay_weight=0.0,
        )
    )

    inputs = jnp.array([[2.0, 3.0, -5.0, 6.0]])

    def loss_fn(params, state):
      del state
      model = SingleLayer(**layer_kwargs)
      y, updated_state = model.apply(
          {
              'params': params,
          },
          inputs,
          update_mask=SparseContext.DynamicContext(update_mask=True),
          apply_mask=True,
          mutable=True,
      )
      total_loss = jnp.sum(y)
      return total_loss, updated_state

    @jax.jit
    def update_params(params, grads):
      params = jax.tree_util.tree_map(lambda p, g: p - g, params, grads)
      return params

    module = SingleLayer(**layer_kwargs)
    variables = module.init(
        rng,
        jnp.zeros(inputs.shape),
        update_mask=SparseContext.DynamicContext(update_mask=False),
        apply_mask=False,
    )
    state, params = variables.pop('params')
    del variables
    for _ in range(10):
      # At each iteration, the pruned weights are multiplied with
      # ste_weight = 0.0002 and added to corresponding gradients.
      # In this simple example, gradients are simply the inputs to the network.
      (_, state), grads = jax.value_and_grad(
          loss_fn, has_aux=True, allow_int=True
      )(params, state)
      np.testing.assert_allclose(
          grads['kernel'],
          inputs
          + 0.0002
          * jnp.multiply(
              ~(state['sparsity']['weight_sparsity']['mask'].astype(bool)),
              params['kernel'],
          ),
      )
      params = update_params(params, grads)


class PruningParamsTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(sparse_type='STRUCTURED_NM', prune_rate=0.1),
      dict(sparse_type='UNSTRUCTURED', prune_rate=(4, 1)),
  )
  def test_invalid_params(self, sparse_type, prune_rate):
    with self.assertRaisesRegex(
        AssertionError, 'prune rate should be either None for no pruning'
    ):
      SparseHParams(type=sparse_type, prune_rate=prune_rate)

  @parameterized.parameters(
      dict(
          sparse_type='STRUCTURED_NM', prune_rate=(4, 1), mask_decay_weight=-0.1
      ),
      dict(sparse_type='UNSTRUCTURED', prune_rate=0.2, mask_decay_weight=-0.1),
  )
  def test_invalid_mask_decay_weight(
      self, sparse_type, prune_rate, mask_decay_weight
  ):
    with self.assertRaisesRegex(
        AssertionError, '.* `mask_decay_weight` must be positive.'
    ):
      SparseHParams(
          type=sparse_type,
          prune_rate=prune_rate,
          mask_decay_weight=mask_decay_weight,
      )

  @parameterized.parameters(
      dict(
          sparse_type='STRUCTURED_NM',
          prune_rate=(4, 1),
          sparse_ste=True,
          mask_decay_weight=0.1,
      ),
      dict(
          sparse_type='UNSTRUCTURED',
          prune_rate=0.2,
          sparse_ste=True,
          mask_decay_weight=0.1,
      ),
  )
  def test_invalid_sparse_ste_with_non_zero_mask_decay_weight(
      self, sparse_type, prune_rate, sparse_ste, mask_decay_weight
  ):
    with self.assertRaisesRegex(
        ValueError, 'SR-STE only works with non-decaying mask.'
    ):
      SparseHParams(
          type=sparse_type,
          prune_rate=prune_rate,
          sparse_ste=sparse_ste,
          mask_decay_weight=mask_decay_weight,
      )

  @parameterized.parameters(
      dict(
          sparse_type='STRUCTURED_NM',
          prune_rate=(4, 1),
          sparse_ste=True,
          structure_decay=True,
      ),
      dict(
          sparse_type='UNSTRUCTURED',
          prune_rate=0.2,
          sparse_ste=True,
          structure_decay=True,
      ),
  )
  def test_invalid_sparse_ste_with_structure_decay(
      self, sparse_type, prune_rate, sparse_ste, structure_decay
  ):
    with self.assertRaisesRegex(
        ValueError, 'SR-STE only works with non-decaying sparse structure.'
    ):
      SparseHParams(
          type=sparse_type,
          prune_rate=prune_rate,
          sparse_ste=sparse_ste,
          structure_decay=structure_decay,
      )

  @parameterized.parameters(
      dict(sparse_type='UNSTRUCTURED', prune_rate=0.2, sparse_ste=True)
  )
  def test_invalid_sparse_ste_with_unstructured_sparsity(
      self, sparse_type, prune_rate, sparse_ste
  ):
    with self.assertRaisesRegex(
        ValueError, 'SR-STE only works with structured sparsity.'
    ):
      SparseHParams(
          type=sparse_type, prune_rate=prune_rate, sparse_ste=sparse_ste
      )

if __name__ == '__main__':
  absltest.main()
