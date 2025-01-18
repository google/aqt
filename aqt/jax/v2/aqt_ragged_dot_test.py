# Copyright 2025 Google LLC
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
"""Correctness tests for aqt_ragged_dot against jax.lax.ragged_dot."""

from typing import Callable

from absl import logging
from aqt.jax.v2 import aqt_ragged_dot
from aqt.jax.v2 import config
import jax
import jax.numpy as jnp
import numpy as np

from google3.testing.pybase import googletest
from google3.testing.pybase import parameterized


def sample_groups(m: int, num_groups: int, key: jax.Array) -> jnp.ndarray:
  # Randomly sample proportions of 'm' that will be assigned to each group.

  # Randomly sample 'num_groups - 1' run ends. The final group will end at 'm'.
  # Sample with replacement so that it's possible to get zero-sized groups.
  ends_no_final = jnp.sort(jax.random.choice(key, m, shape=(num_groups - 1,)))
  ends = jnp.concatenate([ends_no_final, jnp.array([m], dtype=jnp.int32)])

  # Calculate the run starts by shifting ends 1 to the right. The first run
  # starts at zero.
  starts = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), ends_no_final])
  return ends - starts


def random_dense(
    shape: tuple[int, ...],
    key: jax.Array,
    dtype: jnp.dtype,
) -> jnp.ndarray:
  x = jax.random.uniform(key, shape, dtype)
  return x.astype(jnp.bfloat16).astype(dtype)


class AqtRaggedDotTest(parameterized.TestCase):

  def assert_allclose(
      self,
      out: jnp.ndarray,
      expected_out: jnp.ndarray,
      *,
      rtol: float = 1e-5,
      atol: float = 1e-5,
  ):
    """Asserts that two arrays are close.

    The function converts the arrays to float32 before comparing them to
    mitigate the "The DTypes <class 'numpy.dtypes.Float16DType'> and <class
    'numpy.dtype[bfloat16]'> do not have a common DType." error.

    Args:
      out: The actual output.
      expected_out: The expected output.
      rtol: The relative tolerance.
      atol: The absolute tolerance.
    """
    self.assertEqual(out.dtype, expected_out.dtype)
    np.testing.assert_allclose(
        out.astype(jnp.float32),
        expected_out.astype(jnp.float32),
        rtol=rtol,
        atol=atol,
    )

  def make_args(
      self,
      m: int,
      k: int,
      n: int,
      num_groups: int,
      in_dtype: jnp.dtype,
      balanced_groups: bool,
  ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Makes arguments for ragged_dot."""
    key = jax.random.PRNGKey(1234)
    k1, k2, k3 = jax.random.split(key, 3)
    lhs = random_dense((m, k), k1, in_dtype)
    logging.vlog(1, "lhs shape: %s", lhs.shape)
    rhs = random_dense((num_groups, k, n), k2, in_dtype)
    logging.vlog(1, "rhs shape: %s", rhs.shape)
    group_sizes = (
        jnp.full((num_groups,), m // num_groups, dtype=jnp.int32)
        if balanced_groups
        else sample_groups(m, num_groups, k3)
    )
    assert jnp.sum(group_sizes) == m
    logging.vlog(1, "group_sizes shape: %s", group_sizes.shape)
    logging.vlog(1, "group_sizes: %s", group_sizes)
    return lhs, rhs, group_sizes

  @parameterized.product(
      in_dtype=(jnp.bfloat16, jnp.float32),
      balanced_groups=(True, False),
  )
  def test_numeric_correctness(self, in_dtype, balanced_groups):
    m, k, n = 32, 128, 16
    num_groups = 8
    lhs, rhs, group_sizes = self.make_args(
        m, k, n, num_groups, in_dtype, balanced_groups=balanced_groups
    )

    out = jax.lax.ragged_dot(
        lhs,
        rhs,
        group_sizes,
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=in_dtype,
    ).astype(in_dtype)

    actual = aqt_ragged_dot.ragged_dot(lhs, rhs, group_sizes)

    logging.vlog(1, "actual = %s", actual)
    rtol = 1e-2 if in_dtype == jnp.bfloat16 else 5e-3
    self.assert_allclose(actual, out, rtol=rtol)
    if balanced_groups:
      with self.subTest(name="test_correctness_against_dot_general"):
        group_size = m // num_groups
        def reference(
            lhs: jnp.ndarray,
            rhs: jnp.ndarray,
            compute: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        ):
          lhs = jnp.reshape(lhs, (num_groups, group_size, k))
          return compute(lhs, rhs).reshape((m, n))

        dot_dimension_numbers = (([2], [1]), ([0], [0]))
        dot_general = config.config_v4()
        dot_general_compute = lambda lhs, rhs: dot_general(
            lhs, rhs, dimension_numbers=dot_dimension_numbers
        )
        expected = reference(lhs, rhs, dot_general_compute)
        logging.vlog(1, "expected = %s", expected)
        self.assert_allclose(actual, expected, rtol=rtol)

  @parameterized.parameters(list(jax.lax.Precision))
  def test_ragged_dot_with_precision(self, precision):
    lhs, rhs, group_sizes = self.make_args(
        32, 128, 16, 8, jnp.bfloat16, balanced_groups=False
    )
    with self.assertRaises(AssertionError):
      aqt_ragged_dot.ragged_dot(
          lhs,
          rhs,
          group_sizes,
          precision=precision,
      )

if __name__ == "__main__":
  googletest.main()
