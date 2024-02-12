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
"""utils contains all kinds of small pieces of code used throughout AQT.

Code in this file can't depend on any other AQT file.
However, it is acceptable to grow pieces of funcionality in this file and later
promote them to dedicated files.
"""

import functools
import flax.struct
import jax.numpy as jnp

float_dtype = [jnp.bfloat16, jnp.float32, jnp.float64]
int_dtype = [jnp.int4, jnp.int8]


flax_slots_dataclass = functools.partial(
    flax.struct.dataclass, frozen=False, slots=True
)


def static_field():
  return flax.struct.field(pytree_node=False)


def dynamic_field():
  return flax.struct.field(pytree_node=True)
