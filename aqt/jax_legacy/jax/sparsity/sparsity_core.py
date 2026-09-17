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

import functools
from typing import Optional

from aqt.jax_legacy.jax.sparsity import sparsity_hparams
import jax
import jax.numpy as jnp


def apply_sparsity(
    inputs: jnp.ndarray,
    mask: jnp.ndarray,
    channelwise_dim: Optional[int] = None,
) -> jnp.ndarray:
  """Returns sparsified inputs based on input mask.

  Args:
    inputs: The input tensor.
    mask: The mask to be applied to the input tensor.
    channelwise_dim: The dimension in which the masked tensors exist. If
      supplied, the mask will be converted to a channelwise mask and the input
      tensor will be have the masked channels removed, reducing the overall
      size.

  Returns: The masked input tensor.
  """
  if channelwise_dim is not None:
    target_axis = channelwise_dim % inputs.ndim
    non_target_axes = [i for i in range(inputs.ndim) if i != target_axis]
    channel_mask = jnp.any(mask, axis=non_target_axes, keepdims=True)
    channel_mask = channel_mask.reshape(-1)
    return jnp.take(inputs, jnp.nonzero(channel_mask)[0], axis=channelwise_dim)

  return jnp.where(mask, inputs, jnp.zeros(inputs.shape, inputs.dtype))


def prune_channels(
    inputs: jnp.ndarray,
    channelwise_axis: int,
    mask: jnp.ndarray,
) -> jnp.ndarray:
  """Returns pruned inputs based on mask.

  Args:
    inputs: The input tensor.
    channelwise_axis: The axis in which to remove slices/channels.
    mask: The mask to be applied to the input tensor.

  Returns: The pruned inputs with channels removed, reducing the overall
      size.
  """
  output = inputs[mask * jnp.ones_like(inputs).astype(jnp.bool_)]
  mask = mask.squeeze()
  if mask.ndim > 1:
    mask = mask[1]
  num_channels_kept = jnp.sum(mask).item()

  output_shape = list(inputs.shape)
  output_shape[channelwise_axis] = num_channels_kept

  return output.reshape(output_shape)


def get_sparsity_mask(
    inputs: jnp.ndarray,
    n_sparsity: int = 0,
    m_sparsity: int = 0,
    order: str = 'R',
    block_size: int = 0,
    offset: int = 0,
) -> jnp.ndarray:
  """Returns sparsified inputs for n:m structured pruning.

  Args:
    inputs: Input array for which N:M pruning mask is computed.
    n_sparsity: Maximum number of non-zero values in each block.
    m_sparsity: Number of values in each block.
    order: Apply pruning using this index order. Supported values are `C`, `R`.
      `C` and `R` indicate column-wise and row-wise masking, respectively.
      Default is `R` indicating to applying N:M sparsity across rows of the
      input matrix. Default is `C` indicating to applying N:M sparsity across
      columns of the input matrix. The choice may intersect with hardware
      capabilities. For a weight tensor `C` corresponds to the reduction
      dimension, and `R' for activations.
    block_size: Number of values in each weight block.
    offset: Indicates the offset between the group of M elements on which
      N:M sparsity is applied. The default is `0` (narrowly-separated),
        indicating that `M` elements are selected from adjacent values in the
        input matrix. Generally, because of the XLA layout (lanes 128/sublanes
        8), another value for offset would be 128 (widely-separated). If offset
        > 0, we only support scenarios where the input array size is equal to
        (offset * m). Offset != 128 may not be best optimized for the memory
        layout.

  Returns:
    A mask that indicates the pruning locations (`0`: no pruning, `1`: pruned).
  """
  assert (
      n_sparsity <= m_sparsity
  ), f'N must be lower than M for N:M ({n_sparsity}:{m_sparsity}) sparsity.'
  if order not in ['C', 'R']:
    raise ValueError(f'Index order {order} not supported.')
  if offset < 0:
    raise ValueError(f'Offset value must be positive. You provided {offset}.')

  length = jnp.size(inputs)
  if length % m_sparsity != 0:
    raise ValueError(
        f'inputs size must be divisible by m, provided {length} and'
        f' {m_sparsity}'
    )
  if order not in ['C', 'R']:
    raise ValueError(f'Index order {order} not supported.')

  if block_size > 1:
    blocks = int(length / block_size)
    original_shape = inputs.shape
    if order == 'R':
      inputs_block = inputs.reshape(blocks, block_size, order='C')
    else:
      inputs_trans = jnp.einsum('...ij->...ji', inputs)
      original_shape = inputs_trans.shape
      inputs_block = inputs_trans.reshape(blocks, block_size, order='C')

    def block_score(inputs: jnp.ndarray):
      return jnp.sum(jnp.abs(inputs), axis=-1)

    inputs_block_temp = jnp.apply_along_axis(
        block_score, axis=-1, arr=inputs_block
    )
    mask_shape = tuple((
        original_shape[i]
        if i != jnp.size(original_shape) - 1
        else int(original_shape[i] / block_size)
        for i in range(jnp.size(original_shape))
    ))
    if order == 'R':
      new_inputs = inputs_block_temp.reshape(mask_shape, order='C')
    else:
      new_inputs = jnp.einsum(
          '...ij->...ji', inputs_block_temp.reshape(mask_shape, order='C')
      )
    inputs = new_inputs

  length = jnp.size(inputs)
  if offset > 0 and length % (offset * m_sparsity) != 0:
    raise ValueError(
        'When offset > 0, we only support an array size (length) equal to '
        f'(offset * m_sparsity). Provided offset = {offset}, '
        f'm_sparsity = {m_sparsity}, length = {length}.'
    )

  inputs = jnp.abs(inputs)
  original_shape = inputs.shape

  if order == 'C':
    inputs = jnp.einsum('...ij->...ji', inputs)
    original_shape = inputs.shape

  prac_offset = 1 if offset == 0 else offset
  if original_shape[-1] % (m_sparsity * prac_offset) == 0:
    group = original_shape[-1] // m_sparsity
    # TODO(shivaniagrawal): we can always split in 3D with offset=1 too and
    # do top-K in -2 dimension.
    if offset > 1:
      new_shape = (*original_shape[:-1], group // offset, m_sparsity, offset)
      inputs = inputs.reshape(new_shape)
      inputs = jnp.einsum('...ij->...ji', inputs)

    new_shape = (*original_shape[:-1], group, m_sparsity)
    inputs_temp = inputs.reshape(new_shape)

  else:
    group = int(length / m_sparsity)
    if offset > 0:
      inputs = inputs.reshape((group // offset, m_sparsity, offset))
      inputs = jnp.einsum('...ij->...ji', inputs)

    inputs_temp = inputs.reshape(group, m_sparsity, order='C')

  _, top_k_indices = jax.lax.top_k(inputs_temp, k=n_sparsity)
  mask = jnp.any(
      jax.nn.one_hot(top_k_indices, m_sparsity, dtype=jnp.bool_), axis=-2
  )

  if offset > 0:
    # NOTE: without meeting this condition, we had flattened the whole matrix
    # and mask as well.
    if original_shape[-1] % (m_sparsity * offset) == 0:
      # group = original_shape[-1] // m_sparsity in this case
      mask = mask.reshape(
          (*original_shape[:-1], group // offset, offset, m_sparsity)
      )
    else:
      # group = length // m_sparsity in this case
      mask = mask.reshape((group // offset, offset, m_sparsity))
    mask = jnp.einsum('...ij->...ji', mask)

  if order == 'R':
    result_mask = mask.reshape(original_shape, order='C')
  else:
    result_mask = jnp.einsum(
        '...ij->...ji', mask.reshape(original_shape, order='C')
    )

  if block_size > 0:
    if order == 'R':
      expanded_mask = jnp.repeat(result_mask, block_size, axis=-1)
    else:
      expanded_mask = jnp.repeat(result_mask, block_size, axis=-2)
    return expanded_mask
  else:
    return result_mask


@jax.jit
def _topk_mask_calculator_internal(inputs: jnp.ndarray, prune_rate: float):
  """Creates a binary mask given the prune rate on the scores."""
  flat_inputs = jnp.reshape(inputs, -1)
  num_ones = jnp.round(flat_inputs.size * (1 - prune_rate)).astype(int)
  num_ones = jnp.maximum(1, num_ones)

  topk_index = jnp.argsort(-flat_inputs)[num_ones - 1]
  topk_threshold = flat_inputs[topk_index]

  mask_by_value = inputs >= topk_threshold

  # Use lower value indices to prioritize unpruned weight values.
  mask_by_index = (jnp.arange(flat_inputs.size) <= topk_index) | (
      flat_inputs != topk_threshold
  )
  mask_by_index = jnp.reshape(mask_by_index, inputs.shape)
  return mask_by_value * mask_by_index


@functools.partial(jax.jit, static_argnames=['channel_dim'])
def get_sparsity_mask_channelwise(
    inputs: jnp.ndarray, prune_rate: float, channel_dim: int = -1
) -> jnp.ndarray:
  """Returns a mask for the channel-wise pruned input.

  Args:
    inputs: The input matrix to have channel-wise masking applied.
    prune_rate: The rate in which values in the inputs are pruned.
    channel_dim: The channel dimension, the dimension across which channels will
      be pruned.

  Returns:
    A mask that indicates the pruning locations.
  """
  target_axis = channel_dim % inputs.ndim

  non_target_axes = [i for i in range(inputs.ndim) if i != target_axis]
  channel_score = jnp.sum(inputs, axis=non_target_axes, keepdims=True)

  mask = _topk_mask_calculator_internal(channel_score, prune_rate)
  mask = mask * jnp.ones_like(inputs)
  return mask.astype(jnp.bool_)


# TODO(ayazdan): Add support for fast top-k.


def get_sparsity_mask_unstructured(
    inputs: jnp.ndarray,
    mask: jnp.ndarray | None,
    prune_rate: jnp.ndarray | float,
) -> jnp.ndarray:
  """Computes a sparisty mask to prune the required percentage of weights.

  The mask is calculated by thresholding the absolute values of inputs. The
  threshold is the lowest value greater than prune_rate percent of weights, i.e.
  the corresponding percentile.

  The newly pruned weights form a superset of the currently pruned weights if
  the current mask is provided.

  Args:
      inputs: Input tensor.
      mask: Current mask.
      prune_rate: Percentage of weights to prune, value between 0 and 100.

  Returns:
      Sparsity mask.
  """
  if mask is not None:
    inputs = apply_sparsity(inputs, mask)
  inputs_abs = jnp.abs(inputs)
  threshold = jnp.percentile(inputs_abs, prune_rate)
  return jnp.greater(inputs_abs, threshold)


# TODO(shivaniagrawal): Only used for testing the functionality of
# get_prune_mask; update the test to call get_pruning_n_m_mask instead.
def prune_inputs_n_m(
    inputs: jnp.ndarray,
    *,
    n: int,
    m: int,
    order: (
        sparsity_hparams.SparsityOrder | str
    ) = sparsity_hparams.SparsityOrder.C,
    offset: int = 0,
) -> jnp.ndarray:
  """Returns pruned array with N:M (structured) pruning.

  N:M pruning makes at most N values non-zero in each block of M consecutive
  values.

  Args:
    inputs: Input array for which N:M pruning mask is computed.
    n: Maximum number of non-zero values in each block.
    m: Number of values in each block.
    order: Apply pruning using this index order. Supported values are `C`, `R`.
      `C` and `R` indicate column-wise and row-wise masking, respectively.
      Default is `R` indicating to applying N:M sparsity across rows of the
      input matrix. The choice may intersect with hardware capabilities. For a
      weight tensor `C` corresponds to the reduction dimension, and `R' for
      activations.
    offset: Indicates the offset between the group of M elements on which
      N:M sparsity is applied. The default is `0` (narrowly-separated),
        indicating that `M` elements are selected from adjacent values in the
        input matrix. Generally, because of the XLA layout (lanes 128/sublanes
        8), another value for offset would be 128 (widely-separated). If offset
        > 0, we only support scenarios where the input array size is equal to
        (offset * m). Offset != 128 may not be best optimized for the memory
        layout.

  Returns:
    An array with the same shape as inputs pruned with N:M strategy.
  """
  mask = get_sparsity_mask(inputs, n, m, order=order, offset=offset)
  return jnp.where(mask, inputs, jnp.zeros(inputs.shape, inputs.dtype))


SparsityScore = sparsity_hparams.SparsityScore


def compute_score(
    weights: jnp.ndarray,
    score_func: SparsityScore = SparsityScore.MAGNITUDE,
    inputs: jnp.ndarray | None = None,
) -> jnp.ndarray:
  """Compute importance score of weight before pruning."""
  if score_func == SparsityScore.ACTIVATION_WEIGHTED:
    if inputs is None:
      raise ValueError('`inputs` must be given for `ACTIVATION_WEIGHTED`.')
    else:
      return score_activation_weighted(weights, inputs)
  elif score_func == SparsityScore.MAGNITUDE:
    return score_weight_magnitude(weights)
  else:
    raise ValueError('Unknown sparsity score function.')


def score_weight_magnitude(weight: jnp.ndarray) -> jnp.ndarray:  # pylint: disable=unused-argument
  """This function returns score based on the magnitude of weights."""

  return jnp.abs(weight)


def score_activation_weighted(
    weight: jnp.ndarray, inputs: jnp.ndarray
) -> jnp.ndarray:
  """This function returns a weighted score of weights based on the average activation magnitude.

  The score is calculated as the product of the weight magnitude and the mean
  magnitude of the activation tensor.

  Args:
    weight: A 2-D weight matrix of shape (C_in, C_out).
    inputs: A N-D tensor where the last channel is C_in.

  Returns:
    A score with the same shape as weight.
  """

  # TODO(wppark): Add support for attention layers as well.
  if not (jnp.ndim(weight) == 2 and inputs.shape[-1] == weight.shape[0]):
    raise ValueError(
        'ACTIVATION_WEIGHTED score function only supports Linear layers for'
        ' now. Weight must be 2-dimensional matrices, and the last channel of'
        ' inputs must have the same number of dimension of the first channel of'
        ' weight.'
    )
  score = jnp.einsum('...j,jk->jk', jnp.abs(inputs), jnp.abs(weight))
  return score
