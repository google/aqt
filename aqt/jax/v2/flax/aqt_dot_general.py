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
"""Flax layer for AQT injection."""

import functools

from aqt.jax.v2 import aqt_dot_general
from aqt.jax.v2 import config
import flax.linen as nn
import jax.numpy as jnp


class Freezer(nn.Module, config.Preprocess):
  """Stores and reuses values in a variable.

    On default it is an identity function that saves the input in a variable.
  In 'use_frozen=True' mode, ignores the input and returns the frozen value. It
    is usefult to implement 'constant folding' and put quantized weights and
    scales in the checkpoint for serving.
  """

  # If you want use 'params' make sure that there is another mechanism to hide
  # these variables from the optimizer.
  var_collection: str = 'aqt'

  # If you set it to True, instead of returning the current input
  # will return last input it got.
  use_frozen: bool = False

  @nn.compact
  def __call__(self, inputs):
    # return inputs or the frozen value
    frozen = self.variable(self.var_collection, 'val', jnp.zeros, inputs.shape)
    if not self.use_frozen:
      frozen.value = inputs
    return frozen.value


class AqtDotGeneral(nn.Module):
  """A layer that can be injected into flax.nn.Dense, etc."""

  cfg: config.DotGeneral | None = None
  prng_name: str | None = 'params'

  @nn.compact
  def __call__(
      self,
      lhs,
      rhs,
      dimension_numbers,
      precision,
      preferred_element_type=None,
  ):
    key = self.make_rng(self.prng_name) if self.prng_name is not None else None
    context = aqt_dot_general.Context(key=key, train_step=None)
    aqt_dg = aqt_dot_general.make_dot_general(self.cfg)
    aqt_dg = functools.partial(aqt_dg, context=context)
    return aqt_dg(
        lhs,
        rhs,
        dimension_numbers,
        precision,
        preferred_element_type=preferred_element_type,
    )


class AqtEinsum(nn.Module):
  """Quantized Einsum class for model injection."""

  cfg: config.DotGeneral | None = None
  prng_name: str | None = 'params'

  @nn.compact
  def __call__(self, eqn, lhs, rhs):
    key = self.make_rng(self.prng_name) if self.prng_name is not None else None
    context = aqt_dot_general.Context(key=key, train_step=None)
    aqt_dg = aqt_dot_general.make_dot_general(self.cfg)
    aqt_dg = functools.partial(aqt_dg, context=context)
    return jnp.einsum(eqn, lhs, rhs, _dot_general=aqt_dg)
