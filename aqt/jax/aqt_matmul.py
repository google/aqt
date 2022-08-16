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

"""Quantized matmul."""

import builtins

from jax import core
from jax import lax
from jax._src.numpy.util import _check_arraylike
from jax._src.numpy.util import _promote_dtypes
import numpy as np

shape = np.shape
ndim = np.ndim
_max = builtins.max


def matmul(a, b, *, precision=None):  # pylint: disable=missing-docstring
  _check_arraylike("matmul", a, b)
  for i, x in enumerate((a, b)):
    if ndim(x) < 1:
      msg = (f"matmul input operand {i} must have ndim at least 1, "
             f"but it has ndim {ndim(x)}")
      raise ValueError(msg)

  a, b = _promote_dtypes(a, b)

  a_is_mat, b_is_mat = (ndim(a) > 1), (ndim(b) > 1)
  a_batch_dims = shape(a)[:-2] if a_is_mat else ()
  b_batch_dims = shape(b)[:-2] if b_is_mat else ()
  num_batch_dims = _max(len(a_batch_dims), len(b_batch_dims))
  a_batch_dims = (None,) * (num_batch_dims - len(a_batch_dims)) + a_batch_dims
  b_batch_dims = (None,) * (num_batch_dims - len(b_batch_dims)) + b_batch_dims

  # Dimensions to squeeze from the inputs.
  a_squeeze = []
  b_squeeze = []

  # Positions of batch dimensions in squeezed inputs.
  a_batch = []
  b_batch = []

  # Desired index in final output of each kind of dimension, in the order that
  # lax.dot_general will emit them.
  idx_batch = []
  idx_a_other = []  # other = non-batch, non-contracting.
  idx_b_other = []
  for i, (ba, bb) in enumerate(zip(a_batch_dims, b_batch_dims)):
    if ba is None:
      idx_b_other.append(i)
    elif bb is None:
      idx_a_other.append(i)
    elif core.symbolic_equal_dim(ba, 1):
      idx_b_other.append(i)
      a_squeeze.append(len(idx_batch) + len(idx_a_other) + len(a_squeeze))
    elif core.symbolic_equal_dim(bb, 1):
      idx_a_other.append(i)
      b_squeeze.append(len(idx_batch) + len(idx_b_other) + len(b_squeeze))
    elif core.symbolic_equal_dim(ba, bb):
      a_batch.append(len(idx_batch) + len(idx_a_other))
      b_batch.append(len(idx_batch) + len(idx_b_other))
      idx_batch.append(i)
    else:
      raise ValueError("Incompatible shapes for matmul arguments: {} and {}"
                       .format(shape(a), shape(b)))

  if a_is_mat: idx_a_other.append(num_batch_dims)
  if b_is_mat: idx_b_other.append(num_batch_dims + a_is_mat)
  perm = np.argsort(np.concatenate([idx_batch, idx_a_other, idx_b_other]))

  a = lax.squeeze(a, tuple(a_squeeze))
  b = lax.squeeze(b, tuple(b_squeeze))
  out = lax.dot_general(
      a,
      b, (((ndim(a) - 1,), (ndim(b) - 1 - b_is_mat,)), (a_batch, b_batch)),
      precision=precision)
  return lax.transpose(out, perm)
