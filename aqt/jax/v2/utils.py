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
import re
from typing import Any, Sequence
import flax.struct
import jax
from jax import numpy as jnp


# None means that the template matches any axis size
ShapeTemplate = Sequence[int | None]

# TODO(lew): We should get a better type for jax.lax.dot_general and variants.
DotGeneralT = Any


def assert_shape(shape: Sequence[int], shape_template: ShapeTemplate, msg: str):
  shape = list(shape)
  shape_template = list(shape_template)
  assert len(shape) == len(shape_template), msg
  for axis_size, expected_size in zip(shape, shape_template):
    assert (
        expected_size is None or axis_size == expected_size
    ), f'{msg}: {shape} does not match {shape_template}'


def assert_eq(value: Any, expected: Any, value_name: str):
  msg = f'{value_name}={value}, {expected=}'
  assert value == expected, msg


flax_slots_dataclass = functools.partial(
    flax.struct.dataclass, frozen=False, slots=True
)


def static_field(**kwargs):
  return flax.struct.field(pytree_node=False, **kwargs)


def dynamic_field(**kwargs):
  return flax.struct.field(pytree_node=True, **kwargs)


def print_diff(str_a: str, str_b: str):
  diff_generator = difflib.context_diff(str_a.split(' '), str_b.split(' '))
  print('Diff:')
  for diff in diff_generator:
    print(diff)
  print(f'first string (actual):\n{str_a}')


def test_pprint_eq(
    input_a: Any, input_b: Any, remove_memory_addresses: bool = False
):
  str_input_a = input_a if isinstance(input_a, str) else pprint.pformat(input_a)
  str_input_b = input_b if isinstance(input_b, str) else pprint.pformat(input_b)
  if remove_memory_addresses:
    str_input_a = re.sub(r' at 0x.*>', '>', str_input_a, 0, re.MULTILINE)
    str_input_b = re.sub(r' at 0x.*>', '>', str_input_b, 0, re.MULTILINE)
  assert str_input_a == str_input_b, print_diff(str_input_a, str_input_b)


def infer_dtype_from_bits(bits: int, upcast: bool = True) -> jnp.dtype | None:
  """Get the dtype for the number of bits provided.

  Args:
    bits: number of bits for the dtype.
    upcast: if upcasting the return dtype for int4 when platform is cpu.

  Returns:
    The corresponding container dtype for the number of bits provided.
  """
  if bits == 4:
    # this branch should return jnp.int4 directly but
    # lax.dot_general(int4, int4) is illegal on cpu.
    # TODO(aqt): Remove this platform check once
    # https://github.com/google/jax/issues/19682 is fixed.
    if jax.local_devices()[0].platform != 'cpu':
      return jnp.int4
    else:
      # TODO(yichizh): It's better to assert False here with the following msg
      # msg = (
      #     'lax.dot_general(int4, int4) is illegal on cpu:'
      #     ' https://github.com/google/jax/issues/19682. The simple workaround'
      #     ' is to upcast to int8, but in that case please directly set the'
      #     ' numerics bits to int8. Please contact the AQT team if you believe'
      #     ' the workaround is needed.'
      # )
      return jnp.int8 if upcast else jnp.int4
  else:
    if bits <= 8 and bits >= 2:
      return jnp.int8
    else:
      return None


def get_remaining_axes(
    rank: int, contraction_axes: Sequence[int], batch_axes: Sequence[int]):
  """Returns the remaining axes."""
  ret = []
  for i in range(rank):
    if i not in list(contraction_axes) + list(batch_axes):
      ret.append(i)
  return ret
