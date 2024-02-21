# Copyright 2024 Google LLC
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
"""State class to gather running statistics."""

import abc

from aqt.jax.v2 import utils
import jax.numpy as jnp


# 4. The State is a pure data container.
@utils.flax_slots_dataclass
class State(abc.ABC):
  """Contains values of the state."""


@utils.flax_slots_dataclass
class StaticRangeState(State):
  """State for static range AQT/PTQ/QAT."""
  max: jnp.ndarray
