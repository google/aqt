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

import difflib
import functools
import pprint
from typing import Any
import flax.struct


flax_slots_dataclass = functools.partial(
    flax.struct.dataclass, frozen=False, slots=True
)


def static_field(**kwargs):
  return flax.struct.field(pytree_node=False, **kwargs)


def dynamic_field(**kwargs):
  return flax.struct.field(pytree_node=True, **kwargs)


def print_diff(str_a: str, str_b: str):
  diff_generator = difflib.context_diff(str_a.split(' '), str_b.split(' '))
  for diff in diff_generator:
    print(diff)


def test_pprint_eq(input_a: Any, input_b: Any):
  str_input_a = input_a if isinstance(input_a, str) else pprint.pformat(input_a)
  str_input_b = input_b if isinstance(input_b, str) else pprint.pformat(input_b)
  assert str_input_a == str_input_b, print_diff(str_input_a, str_input_b)
