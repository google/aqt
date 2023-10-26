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


class AqtDotGeneral(nn.Module):
  """A layer that can be injected into flax.nn.Dense, etc."""

  aqt_cfg: config.DotGeneral | None = None
  track_train_step: bool = False

  @nn.compact
  def __call__(
      self,
      lhs,
      rhs,
      dimension_numbers,
      precision,
      preferred_element_type=None,
  ):
    aqt_key = self.make_rng('params')
    context = aqt_dot_general.Context(key=aqt_key, train_step=None)
    aqt_dg = aqt_dot_general.make_dot_general(self.aqt_cfg)
    aqt_dg = functools.partial(aqt_dg, context=context)
    return aqt_dg(
        lhs,
        rhs,
        dimension_numbers,
        precision,
        preferred_element_type=preferred_element_type,
    )
