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

"""Basic functionalities for pruning neural networks implemented in jax."""

import dataclasses
import enum
import functools
import math
import typing
from typing import Tuple, Union

from absl import logging
from aqt.jax_legacy.jax.flax import struct as flax_struct
import aqt.jax_legacy.jax.sparse_context as SparseContext
from flax import linen as nn
from flax.linen import partitioning
import jax
import jax.numpy as jnp


dataclass = (
    flax_struct.dataclass if not typing.TYPE_CHECKING else dataclasses.dataclass
)


# TODO(ayazdan): Create abstract class for sparsity configurations.
class SparseType(str, enum.Enum):
  """Pruning types dataclass."""

  STRUCTURED_NM = 'STRUCTURED_NM'
  UNSTRUCTURED = 'UNSTRUCTURED'
  STRUCTURED_NMC = 'STRUCTURED_NMC'


@dataclass
class SparseHParams:
  """Hyper parameters for sparsity.

  Attributes:
    type: Input array for which pruning mask is computed.
    prune_rate: Pruning rate.
    order: Apply pruning using this index order. Supported values are `C`, `R`.
      `C` and `R` indicate column-wise and row-wise ordering, respectively.
    absolute: If True, the absolute value of the values are used for sorting.
    smallest: If True, the smallest values in inputs are masked.
    structure_decay: If True, a decaying mechanism is applied on the structure.
    mask_decay_weight: If 0.0, no mask decay is applied. The mask value
      start with 1.0 and each time `num_update_sparsity` * `mask_decay_weight`
      is subtracted from 1.0. Due to overhead of jit, we limited the number of
      updates to `num_update_sparsity` to 16. After 16 iterations, we forcefully
      set `mask_decay_value` to zero. Mask decaying works for both structured
      and unstructured sparsity.
    sparse_ste: If True, a sparse-refined straight-through estimator (SR-STE)
      is applied, following the algorithm described in:
        https://arxiv.org/abs/2102.04010
    sparse_ste_weight: Denotes the relative weight for the sparse-refined term.
      As mentioned in the paper (https://arxiv.org/abs/2102.04010), the best
      default value is 0.0002 (lambda_w in the paper).
    update_mask_frequency: Indicates how often the mask should be updated.
    counter: use the counter
  """

  type: SparseType
  # Prune_rate data type varies with SparseHParams::type.
  # float: If SparseHParams::type is UNSTRUCTURED.
  # Tuple[int]: If SparseHParams::type is STRUCTURED_NM,  Tuple of type N, M.
  prune_rate: Union[None, float, Tuple[int, int], Tuple[int, int, int]]
  order: str = 'R'
  absolute: bool = True
  smallest: bool = True
  structure_decay: bool = False
  mask_decay_weight: float = 0.0
  sparse_ste: bool = False
  sparse_ste_weight: float = 0.0002
  update_mask_frequency: int = 10
  counter: int = 0

  def __post_init__(self):
    if self.prune_rate is not None:
      if self.type == SparseType.STRUCTURED_NM:
        assert isinstance(self.prune_rate, Tuple), (
            'prune rate should be either '
            'None for no pruning or a '
            'Tuple (N, M) for '
            'STRUCTURED_NM sparsity'
        )
      elif self.type == SparseType.UNSTRUCTURED:
        assert isinstance(self.prune_rate, float), (
            'prune rate should be either '
            'None for no pruning or float for '
            'UNSTRUCTURED sparsity'
        )
      elif self.type == SparseType.STRUCTURED_NMC:
        assert isinstance(self.prune_rate, Tuple), (
            'prune rate should be either '
            'None for no pruning or a '
            'Tuple (N, M, C) for STRUCTURED_NMC (N:MxC) sparsity'
        )
      else:
        assert False, 'prune rate unknown!'

      assert self.mask_decay_weight >= 0.0, (
          'Invalid value for '
          f'{self.mask_decay_weight}. '
          '`mask_decay_weight` must be positive.'
      )

      if self.sparse_ste:
        if self.mask_decay_weight != 0.0:
          raise ValueError('SR-STE only works with non-decaying mask.')
        if self.structure_decay:
          raise ValueError(
              'SR-STE only works with non-decaying sparse structure.'
          )
        if self.type != SparseType.STRUCTURED_NM:
          raise ValueError('SR-STE only works with structured sparsity.')


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def sr_ste(
    inputs: jnp.ndarray,
    mask: jnp.ndarray,
    update_mask: bool,
    apply_mask: bool,
    sparsity_hparams: SparseHParams,
    n_sparsity: int = 0,
    m_sparsity: int = 0,
):
  """Wrapper function for custom derivative rule for structured sparsity.

  Algorithm description: https://arxiv.org/abs/2102.04010

  The last three arguments are forced to be static to simplify
    the implementation.

  Args:
    inputs: Input array for which N:M pruning mask is computed.
    mask: The mask matrix which defines which elements to be pruned.
    update_mask: If True, the mask pattern gets updated.
    apply_mask: If True, the mask is applied to input.
    sparsity_hparams: The hyperparmeters related to sparsity.
    n_sparsity: Integer value for N in N:M sparsity.
    m_sparsity: Integer value for M in N:M sparsity.

  Returns:
    The updated input values after applying sparsity.
  """

  return sr_ste_fwd(
      inputs=inputs,
      mask=mask,
      update_mask=update_mask,
      apply_mask=apply_mask,
      sparsity_hparams=sparsity_hparams,
      n_sparsity=n_sparsity,
      m_sparsity=m_sparsity,
  )[0]


@functools.partial(jax.jit, static_argnums=(4, 5, 6))
def sr_ste_fwd(
    inputs: jnp.ndarray,
    mask: jnp.ndarray,
    update_mask: bool,
    apply_mask: bool,
    sparsity_hparams: SparseHParams,
    n_sparsity: int = 0,
    m_sparsity: int = 0,
) -> jnp.ndarray:
  """Custom forward pass for structured sparsity."""
  # pylint:disable=g-long-lambda
  updated_mask = jax.lax.cond(
      update_mask,
      lambda: get_sparsity_mask(
          inputs, sparsity_hparams, n_sparsity, m_sparsity
      ),
      lambda: mask,
  )
  updated_inputs = jax.lax.cond(
      apply_mask, lambda: jnp.multiply(updated_mask, inputs), lambda: inputs
  )
  # pylint:enable=g-long-lambda
  return (
      updated_inputs,
      updated_mask,  # pytype: disable=bad-return-type  # jax-ndarray
      jnp.array(SparseHParams.sparse_ste_weight),
  ), (inputs, updated_mask, jnp.array(SparseHParams.sparse_ste_weight))


def sr_ste_bwd(sparsity_hparams, n_sparsity, m_sparsity, res, g):
  """Implements custom gradient for backward pass.

  Args:
    sparsity_hparams: Non-diff arguments as defined in `sr_ste`.
    n_sparsity: Non-diff arguments as defined in `sr_ste`.
    m_sparsity: Non-diff arguments as defined in `sr_ste`.
    res: Residuals computed in sr_ste_fwd.
    g: Default calculated gradients.

  Returns:
    Gradients for differentiable inputs:
      - inputs
      - mask
      - update_mask
      - apply_mask
  """
  del sparsity_hparams, n_sparsity, m_sparsity
  inputs, updated_mask, ste_weight = res
  # g contains a list of gradients, one per output.
  # g1: updated_inputs
  g1, _, _ = g
  g1 = g1 + ste_weight * jnp.multiply(~(updated_mask.astype(bool)), inputs)
  return (g1, None, None, None)


sr_ste.defvjp(sr_ste_fwd, sr_ste_bwd)


class Sparsity(nn.Module):
  """Abstract class sparsity for applying sparsity."""

  sparsity_hparams: SparseHParams

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      *,
      apply_mask: bool,
      update_mask: SparseContext.DynamicContext = SparseContext.DynamicContext(
          update_mask=False
      ),
      num_update_sparsity: int = 0,
      kernel_axis_names=None,
  ) -> jnp.ndarray:
    if (
        self.sparsity_hparams is None
        or self.sparsity_hparams.prune_rate is None
    ):
      return inputs

    if self.sparsity_hparams.type == 'STRUCTURED_NM':
      n_sparsity = self.sparsity_hparams.prune_rate[0]
      m_sparsity = self.sparsity_hparams.prune_rate[1]
      c_sparsity = 0  # this will be ignored.
      if self.sparsity_hparams.structure_decay:
        if num_update_sparsity == 1:
          n_sparsity = n_sparsity - 1
        else:
          n_sparsity = int(
              math.ceil(n_sparsity / math.pow(2, num_update_sparsity))
          )
    elif self.sparsity_hparams.type == 'STRUCTURED_NMC':
      n_sparsity = self.sparsity_hparams.prune_rate[0]
      m_sparsity = self.sparsity_hparams.prune_rate[1]
      c_sparsity = self.sparsity_hparams.prune_rate[2]
    else:
      logging.info('Unstructured sparsity does not support structure decaying.')
      n_sparsity = 0
      m_sparsity = 0

    mask_decay_value = 1.0
    if self.sparsity_hparams.mask_decay_weight != 0.0:
      if num_update_sparsity < 16:
        mask_decay_value = max(
            mask_decay_value
            - (num_update_sparsity * self.sparsity_hparams.mask_decay_weight),
            0.0,
        )
      else:
        mask_decay_value = 0.0

    mask = partitioning.variable_with_axes(
        'sparsity',
        'mask',
        init_fn=lambda: jnp.zeros_like(inputs, dtype=inputs.dtype),
        axes=kernel_axis_names,
    )
    if self.sparsity_hparams.sparse_ste:
      updated_inputs, updated_mask, _ = sr_ste(
          inputs=inputs,
          mask=mask.value,
          update_mask=update_mask.update_mask,
          apply_mask=apply_mask,
          sparsity_hparams=self.sparsity_hparams,
          n_sparsity=n_sparsity,
          m_sparsity=m_sparsity)
      if update_mask.update_mask and self.has_variable('sparsity', 'mask'):
        mask.value = updated_mask
      return updated_inputs

    if update_mask.update_mask and self.has_variable('sparsity', 'mask'):
      mask.value = get_sparsity_mask(
          inputs, self.sparsity_hparams, n_sparsity, m_sparsity, c_sparsity
      )
    if apply_mask and self.has_variable('sparsity', 'mask'):
      if self.sparsity_hparams.mask_decay_weight != 0.0:
        return jnp.multiply(
            ~(mask.value.astype(bool)) * mask_decay_value + mask.value, inputs
        ).astype(inputs.dtype)
      else:
        return jax.numpy.multiply(mask.value, inputs).astype(inputs.dtype)
    return inputs


def get_sparsity_mask(
    inputs: jnp.ndarray,
    sparsity_hparams: SparseHParams,
    n_sparsity: int = 0,
    m_sparsity: int = 0,
    c_sparsity: int = 0,
):
  """Returns sparsified inputs based on sparsity hparams."""
  if sparsity_hparams is None or sparsity_hparams.prune_rate is None:
    return jnp.ones(inputs.shape, dtype=inputs.dtype)
  prune_rate = sparsity_hparams.prune_rate
  if sparsity_hparams.type == 'STRUCTURED_NM':
    assert isinstance(
        prune_rate, Tuple
    ), 'prune rate must be tuple for structured sparsity.'
    assert prune_rate[0] <= prune_rate[1], (
        'prune_rate[0] must be lower than prune_rate[1] for N:M'
        f' ({prune_rate[0]}:{prune_rate[1]}) sparsity.'
    )
    return get_pruning_n_m_mask(inputs, n=n_sparsity, m=m_sparsity)
  elif sparsity_hparams.type == 'STRUCTURED_NMC':
    return get_pruning_n_m_c_mask(
        inputs, n=n_sparsity, m=m_sparsity, c=c_sparsity
    )
  elif sparsity_hparams.type == 'UNSTRUCTURED':
    assert (
        isinstance(prune_rate, float) and prune_rate < 1
    ), f'sparsity ratio can not be > 1, provided prune_rate {prune_rate}.'
    return get_pruning_unstruct_mask(inputs, prune_rate=prune_rate)
  else:
    raise ValueError(f'invalid sparsity type {sparsity_hparams.type}!')


def get_pruning_n_m_mask(
    inputs: jnp.ndarray,
    n: int,
    m: int,
) -> jnp.ndarray:
  """Returns a mask for N:M (structured) pruning.

  N:M pruning makes at most N values non-zero in each block of M consecutive
  values.

  Args:
    inputs: Input array for which N:M pruning mask is computed.
    n: Maximum number of non-zero values in each block.
    m: Number of values in each block.

  Returns:
    A mask that indicates the pruning locations (`0`: no pruning, `1`: pruned).
  """
  if n > m:
    raise ValueError(f'N must be lower than M for N:M ({n}:{m}) sparsity.')
  length = jnp.size(inputs)
  # TODO(b/228458062): support m which is not a factor of inputs size.
  if length % m != 0:
    raise ValueError(
        f'inputs size must be divisible by m, provided {length} and {m}'
    )
  group = int(length / m)
  inputs = jnp.abs(inputs)
  inputs_temp = inputs.reshape(group, m, order='C')
  ranks = inputs_temp.argsort().argsort()
  mask = (ranks >= m - n).astype(inputs.dtype)
  return mask.reshape(inputs.shape, order='C')


def get_pruning_n_m_c_mask(
    inputs: jnp.ndarray,
    n: int,
    m: int,
    c: int,
) -> jnp.ndarray:
  """Returns a mask for N:MxC structured pruning.

  N:MxC pruning makes at most N values non-zero in each block of MxC.

  Args:
    inputs: Input array for which N:M pruning mask is computed.
    n: Maximum number of non-zero values in each block.
    m: Number of values in each block along rows.
    c: Number of values in each block along columns.

  Returns:
    A mask that indicates the pruning locations (`0`: pruned, `1` not pruned)
  """
  if n > m * c:
    raise ValueError(
        f'N must be lower than MxC for N:MxC ({n}:{m}x{c}) sparsity.'
    )
  col = inputs.shape[1] // c

  inputs = jnp.abs(inputs)
  input_temp = inputs.reshape(-1, m, col, c).swapaxes(1, 2).reshape(-1, m * c)

  mask = jnp.zeros(input_temp.shape, dtype=inputs.dtype)
  _, top_k_indices = jax.lax.top_k(input_temp, k=n)
  mask = jax.vmap(lambda x, i: x.at[i].set(1))(mask, top_k_indices)
  return (
      mask.reshape(-1, col, m, c)
      .swapaxes(1, 2)
      .reshape(inputs.shape[0], inputs.shape[1])
  )


def get_pruning_unstruct_mask(
    inputs: jnp.ndarray, *, prune_rate: float = 0.1
) -> jnp.ndarray:
  """Returns a mask for pruning according to the prune rate.

  Args:
    inputs: Input array for which pruning mask is computed.
    prune_rate: Pruning rate. The ratio of the elements that are pruned. 0
      meaning no pruning. Defaults to 0.1.

  Returns:
    A mask that indicates the pruning locations.
  """
  inputs = jnp.abs(inputs)
  mask = jnp.zeros(inputs.shape, dtype=inputs.dtype)
  k = int(inputs.size * (1.0 - prune_rate))
  _, idxs = jax.lax.top_k(inputs.reshape(-1), k)
  return mask.reshape(-1).at[idxs].set(1.0).reshape(inputs.shape)
