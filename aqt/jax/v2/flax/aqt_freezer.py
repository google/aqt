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
"""Freezers."""

import abc
from typing import Iterable, Optional

from aqt.jax.v2 import aqt_tensor
from aqt.jax.v2.flax import aqt_flax_constant
import flax.linen as nn
from jax import numpy as jnp


_QuantMode = aqt_flax_constant.QuantMode


class AbstractFreezer(abc.ABC, nn.Module):
  """Abstract Identity function that can freeze its inputs."""
  quant_collection: str
  quant_mode: aqt_flax_constant.QuantMode
  q_shape: Iterable[int]
  q_dtype: jnp.dtype
  q_init: nn.initializers.Initializer
  s_shape: Iterable[int]
  s_init: nn.initializers.Initializer

  @abc.abstractmethod
  def get(self) -> Optional[aqt_tensor.QTensor]:
    pass

  @abc.abstractmethod
  def set(self, inputs: aqt_tensor.QTensor) -> None:
    pass


# Leave the class name as Freezer to prevent any side effect for the already-
# existing checkpoints.
class Freezer(AbstractFreezer):
  """Identity function that can freeze its input.

  On default it is an identity function that saves the input in a variable.
  In 'quant_mode=QuantMode.Serve' mode, ignores the input and returns the frozen
  value. It is usefult to implement 'constant folding' and put quantized weights
  and scales in the checkpoint for serving. Specifically:

  self.get() returns None when quant_mode=TRAIN or CONVERT, returns variable
  when quant_mode=SERVE.
  self.set() does nothing when quant_mode=TRAIN or SERVE, creates and stores
  quantized tensor when quant_mode=CONVERT.
  """

  def setup(self):
    mode = self.quant_mode
    if mode == _QuantMode.SERVE or mode == _QuantMode.CONVERT:
      collection = self.quant_collection
      q_init = self.q_init
      q_shape = self.q_shape
      q_dtype = self.q_dtype
      s_init = self.s_init
      s_shape = self.s_shape
      # TODO(lew): Store whole QTensor?
      # We could have created one self.variable whose value is a QTensor,
      # but we are unsure how this would complicate the init function,
      # which could potentially be used by adding metadata such as
      # sharding axises, etc.
      self.qvalue = self.variable(collection, 'value', q_init, q_shape, q_dtype)
      self.scale_t = self.variable(collection, 'scale', s_init, s_shape)

  def get(self) -> Optional[aqt_tensor.QTensor]:
    if self.quant_mode == _QuantMode.TRAIN:
      return None
    elif self.quant_mode == _QuantMode.CONVERT:
      return None
    elif self.quant_mode == _QuantMode.SERVE:
      qvalue = self.qvalue.value
      # TODO(b/325626080): Remove the optional logic.
      if self.q_dtype == jnp.int4:
        qvalue = qvalue.astype(jnp.int4)
      return aqt_tensor.QTensor(
          qvalue,
          scale=None,
          scale_t=[self.scale_t.value],
          dequant_dtype=None,  # Rely on dg output dtype for dequant
      )
    else:
      assert False, 'Unknown quant mode.'

  def set(self, inputs: aqt_tensor.QTensor) -> None:
    # TODO(b/325626080): Uncomment the assert.
    # assert inputs.qvalue.dtype == self.q_dtype, (
    #     f'Freezer got a QTensor of type {inputs.qvalue.dtype} but expected'
    #     f' {self.q_dtype}.'
    # )
    if self.quant_mode == _QuantMode.TRAIN:
      pass
    elif self.quant_mode == _QuantMode.CONVERT:
      qvalue = inputs.qvalue
      # TODO(b/325626080): Remove the optional logic.
      if self.q_dtype == jnp.int4:
        assert qvalue.dtype == jnp.int4
        qvalue = qvalue.astype(jnp.int8)

      self.qvalue.value = qvalue
      assert inputs.scale_t is not None and len(inputs.scale_t) == 1
      self.scale_t.value = inputs.scale_t[0]
    elif self.quant_mode == _QuantMode.SERVE:
      # TODO(lew): Optionally compare stored and served value.
      pass
    else:
      assert False, 'Unknown quant mode.'
    return None
