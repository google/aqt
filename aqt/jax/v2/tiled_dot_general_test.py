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

"""Test for AQT Tiled Dot General."""
import itertools

from absl.testing import absltest
from absl.testing import parameterized
from aqt.jax.v2 import tiled_dot_general
from aqt.jax.v2 import utils
import jax
import jax.numpy as jnp
import numpy as np
import sympy


AxisIdx = utils.AxisIdx
TensorTiling = tiled_dot_general.TensorTiling
Cfg = tiled_dot_general.Cfg
AxisTiling = tiled_dot_general.AxisTiling


def get_shape_from_axes(axes: list[AxisIdx], shape) -> list[int]:
  return [shape[i] for i in axes]


def assign_input_shape(
    rng_key, ca_shape: list[int], ba_shape: list[int], ra_shape: list[int]
):
  """Randomly assign an axis to a shape."""
  num_ca = len(ca_shape)
  num_ba = len(ba_shape)
  num_ra = len(ra_shape)
  total_num_axes = num_ca + num_ba + num_ra
  axes = jax.random.permutation(rng_key, total_num_axes).tolist()
  ca = axes[:num_ca]
  ba = axes[num_ca : num_ca + num_ba]
  ra = axes[num_ca + num_ba :]
  raw_input_shape = np.array(ca_shape + ba_shape + ra_shape)
  input_shape = raw_input_shape[np.argsort(axes)].tolist()
  return ca, ba, ra, input_shape


def get_axis_tiles(
    axes: list[AxisIdx], shape: list[int]
) -> list[list[AxisTiling]]:
  # Assume tiling all axes
  ret = list(
      itertools.product(*[
          [
              tiled_dot_general.AxisTiling(axis, tc, shape[axis] // tc)
              for tc in sympy.divisors(shape[axis])
          ]
          for axis in axes
      ])
  )
  return ret


def generate_inputs(
    rng_key,
    num_ca: int,
    num_ba: int,
    num_lhs_ra: int,
    num_rhs_ra: int,
    max_shape_val: int,
):
  keys = jax.random.split(rng_key, 6)

  ca_shape = jax.random.randint(
      keys[0], shape=(num_ca,), minval=1, maxval=max_shape_val
  ).tolist()
  ba_shape = jax.random.randint(
      keys[1], shape=(num_ba,), minval=1, maxval=max_shape_val
  ).tolist()
  lhs_ra_shape = jax.random.randint(
      keys[2], shape=(num_lhs_ra,), minval=1, maxval=max_shape_val
  ).tolist()
  rhs_ra_shape = jax.random.randint(
      keys[3], shape=(num_rhs_ra,), minval=1, maxval=max_shape_val
  ).tolist()

  lhs_ca, lhs_ba, lhs_ra, lhs_shape = assign_input_shape(
      keys[4], ca_shape, ba_shape, lhs_ra_shape
  )
  rhs_ca, rhs_ba, rhs_ra, rhs_shape = assign_input_shape(
      keys[5], ca_shape, ba_shape, rhs_ra_shape
  )
  assert get_shape_from_axes(lhs_ca, lhs_shape) == get_shape_from_axes(
      rhs_ca, rhs_shape
  ), 'Found lhs ca and rhs ca shapes are different.'
  assert get_shape_from_axes(lhs_ba, lhs_shape) == get_shape_from_axes(
      rhs_ba, rhs_shape
  ), 'Found lhs ba and rhs ba shapes are different.'

  lhs = jnp.ones(lhs_shape)
  rhs = jnp.ones(rhs_shape)
  return lhs, rhs, (lhs_ca, lhs_ba, lhs_ra), (rhs_ca, rhs_ba, rhs_ra)


def generate_tiling_cfgs(lhs_shape, lhs_ca, lhs_ra, rhs_shape, rhs_ca, rhs_ra):
  lhs_remaining_axis_tilings = get_axis_tiles(lhs_ra, lhs_shape)
  lhs_contraction_axes_tilings = get_axis_tiles(lhs_ca, lhs_shape)
  rhs_remaining_axis_tilings = get_axis_tiles(rhs_ra, rhs_shape)
  rhs_contraction_axes_tilings = get_axis_tiles(rhs_ca, rhs_shape)

  cfgs = []
  assert len(lhs_contraction_axes_tilings) == len(rhs_contraction_axes_tilings)
  # Skip some tests since test size is too big
  for i in range(0, len(lhs_contraction_axes_tilings), 3):
    for j in range(0, len(lhs_remaining_axis_tilings), 3):
      for k in range(0, len(rhs_remaining_axis_tilings), 3):
        cfgs.append(
            tiled_dot_general.Cfg(
                lhs=tiled_dot_general.TensorTiling(
                    contraction_axes=list(lhs_contraction_axes_tilings[i]),
                    remaining_axes=list(lhs_remaining_axis_tilings[j]),
                ),
                rhs=tiled_dot_general.TensorTiling(
                    contraction_axes=list(rhs_contraction_axes_tilings[i]),
                    remaining_axes=list(rhs_remaining_axis_tilings[k]),
                ),
            )
        )
  print(f'{len(cfgs)} configs in total.')
  return cfgs


class TiledDotGeneralTest(parameterized.TestCase):

  def test_tiled_dot_general(self):
    num_ca = 2
    num_ba = 2
    num_lhs_ra = 2
    num_rhs_ra = 2
    max_shape_val = 32
    key = jax.random.PRNGKey(7)
    lhs, rhs, lhs_axes, rhs_axes = generate_inputs(
        key, num_ca, num_ba, num_lhs_ra, num_rhs_ra, max_shape_val
    )
    lhs_ca, lhs_ba, lhs_ra = lhs_axes
    rhs_ca, rhs_ba, rhs_ra = rhs_axes
    dims = ((tuple(lhs_ca), tuple(rhs_ca)), (tuple(lhs_ba), tuple(rhs_ba)))
    expected_output = jax.lax.dot_general(lhs, rhs, dims)

    tiling_cfgs = generate_tiling_cfgs(
        lhs.shape, lhs_ca, lhs_ra, rhs.shape, rhs_ca, rhs_ra
    )
    for i in range(len(tiling_cfgs)):
      cfg_in = tiling_cfgs[i]
      output = tiled_dot_general.tiled_dot_general(cfg_in, lhs, rhs, dims)
      msg = (
          'Test failed. Please try the following single test:\n'
          f'lhs_shape = {lhs.shape}\n'
          f'rhs_shape = {rhs.shape}\n'
          'lhs = jnp.ones(lhs_shape)\n'
          'rhs = jnp.ones(rhs_shape)\n'
          f'cfg = {cfg_in}\n'
          f'dims = {dims}\n'
          'out = tiled_dot_general.local_dg(cfg, lhs, rhs, dims)\n'
          'assert out.shape == jax.lax.dot_general(lhs, rhs, dims).shape'
      )
      # assert output.shape == expected_output.shape, msg
      assert (output == expected_output).all(), msg

  def test_single(self):
    lhs_shape = (31, 22, 9, 22, 27)
    rhs_shape = (22, 9, 22, 27, 7, 12)
    lhs = jnp.ones(lhs_shape)
    rhs = jnp.ones(rhs_shape)
    cfg = Cfg(
        lhs=TensorTiling(
            contraction_axes=[
                AxisTiling(axis=2, tile_count=1, tile_size=9),
                AxisTiling(axis=1, tile_count=1, tile_size=22),
            ],
            remaining_axes=[AxisTiling(axis=0, tile_count=1, tile_size=31)],
        ),
        rhs=TensorTiling(
            contraction_axes=[
                AxisTiling(axis=1, tile_count=1, tile_size=9),
                AxisTiling(axis=2, tile_count=1, tile_size=22),
            ],
            remaining_axes=[
                AxisTiling(axis=5, tile_count=1, tile_size=12),
                AxisTiling(axis=4, tile_count=7, tile_size=1),
            ],
        ),
    )
    dims = (((2, 1), (1, 2)), ((4, 3), (3, 0)))
    out = tiled_dot_general.tiled_dot_general(cfg, lhs, rhs, dims)
    assert out.shape == jax.lax.dot_general(lhs, rhs, dims).shape


if __name__ == '__main__':
  absltest.main()
