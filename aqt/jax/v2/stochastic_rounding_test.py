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

"""Test for stochastic rounding."""

from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax.v2 import stochastic_rounding
import jax
from numpy import testing as np_testing


JaxUniform = stochastic_rounding.JaxUniform
RandomCenteredUniform = stochastic_rounding.RandomCenteredUniform


class StochasticRoundingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("jax_uniform", JaxUniform()),
      ("random_centered_uniform", RandomCenteredUniform()),
  )
  def test_range(self, noise_fn):
    noises = noise_fn(shape=(10000,), key=jax.random.PRNGKey(0))
    np_testing.assert_array_less(noises, 0.5)
    np_testing.assert_array_less(-0.5, noises)

  @parameterized.named_parameters(
      ("jax_uniform", JaxUniform()),
      ("random_centered_uniform", RandomCenteredUniform()),
  )
  def test_shape(self, noise_fn):
    noise_sharing_axes = (0,)
    noises = noise_fn(
        shape=(2, 3, 4),
        key=jax.random.PRNGKey(0),
        noise_sharing_axes=noise_sharing_axes,
    )
    self.assertEqual(noises.shape, (1, 3, 4))


if __name__ == "__main__":
  absltest.main()
