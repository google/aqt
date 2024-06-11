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
"""General dot_general tiling."""

# Lingo in this file:
#
# - lhs(rhs) - left(right) hand side of a binary operation
# - ca - contraction axes
# - ba - batch axes
# - ra - remaining axes

# pylint: disable=g-explicit-bool-comparison
# pylint: disable=g-explicit-length-test

import copy
import dataclasses
import pprint
from typing import Iterable
from absl import logging
from aqt.jax.v2 import utils
import jax
import jax.numpy as jnp
from typing_extensions import Self  # for python version < 3.11


AxisIdx = utils.AxisIdx
AxisSize = utils.AxisSize
EinsumEqnLetter = str
EinsumTileSizes = dict[EinsumEqnLetter, AxisSize]

BROADCAST_PREFIX = 'broadcast_'


@dataclasses.dataclass(frozen=False, slots=True)
class AxisTiling:
  axis: AxisIdx
  # At most one of tile_count, tile_size can be None.
  tile_count: AxisSize | None
  tile_size: AxisSize | None


@dataclasses.dataclass(frozen=False, slots=True)
class TensorTiling:
  contraction_axes: list[AxisTiling]
  remaining_axes: list[AxisTiling]
  # There is no point in tiling dot_general's batch axes


@dataclasses.dataclass(frozen=False, slots=True)
class Cfg:
  """Sequence of (lhs, rhs) configurations."""
  lhs: TensorTiling
  rhs: TensorTiling

  @classmethod
  def from_einsum(cls, eqn: str, einsum_tile_sizes: EinsumTileSizes) -> Self:
    """Creates Cfg based on einsum equation and tile sizes."""
    args_eqn, ret_eqn1 = eqn.split('->')
    args_eqn = args_eqn.split(',')
    assert len(args_eqn) == 2, f'more then 2 arguments are not supported {eqn}'
    lhs_eqn1, rhs_eqn1 = args_eqn
    assert lhs_eqn1.isalpha(), f'unsupported lhs: {eqn}'
    assert rhs_eqn1.isalpha(), f'unsupported rhs: {eqn}'
    assert ret_eqn1.isalpha(), f'unsupported ret: {eqn}'
    ret = Cfg(
        lhs=TensorTiling(contraction_axes=[], remaining_axes=[]),
        rhs=TensorTiling(contraction_axes=[], remaining_axes=[]),
    )
    for einsum_letter, tile_size in einsum_tile_sizes.items():
      assert einsum_letter.isalpha()
      assert len(einsum_letter) == 1
      in_lhs = lhs_eqn1.find(einsum_letter)
      in_rhs = rhs_eqn1.find(einsum_letter)
      in_ret = ret_eqn1.find(einsum_letter)
      found_in_lhs = in_lhs != -1
      found_in_rhs = in_rhs != -1
      found_in_ret = in_ret != -1

      def make_tiling(axis, tile_size):
        return AxisTiling(axis=axis, tile_size=tile_size, tile_count=None)

      msg = (
          'We support only contraction axes and remaining axes:'
          f' eqn={eqn} einsum_letter={einsum_letter}'
      )
      assert found_in_lhs + found_in_rhs + found_in_ret == 2, msg
      if found_in_lhs and found_in_rhs:
        # Contraction axis
        ret.lhs.contraction_axes.append(make_tiling(in_lhs, tile_size))
        ret.rhs.contraction_axes.append(make_tiling(in_rhs, tile_size))
      else:
        # Remaining axis
        assert found_in_ret, f'Should not happen. {msg}'
        if found_in_lhs:
          ret.lhs.remaining_axes.append(make_tiling(in_lhs, tile_size))
        if found_in_rhs:
          ret.rhs.remaining_axes.append(make_tiling(in_rhs, tile_size))
    return ret

  def complete_missing(
      self,
      lhs_shape: tuple[AxisSize, ...],
      rhs_shape: tuple[AxisSize, ...],
      dimension_numbers: jax.lax.DotDimensionNumbers,
  ) -> Self:
    """Makes lhs and rhs to cover all the axes."""
    new_cfg = copy.deepcopy(self)
    (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
    lhs_ra = get_ra(len(lhs_shape), lhs_ca, lhs_ba)
    rhs_ra = get_ra(len(rhs_shape), rhs_ca, rhs_ba)

    def f(axes_cfg, axes, shape):
      # add missing tile_count
      for axis_tiling in axes_cfg:
        tc = axis_tiling.tile_count
        ts = axis_tiling.tile_size
        axis_shape = shape[axis_tiling.axis]
        msg = 'At most one of tile_count and tile_size can be None'
        assert not (tc is None and ts is None), msg
        axis_tiling.tile_count = axis_shape // ts if tc is None else tc
        axis_tiling.tile_size = axis_shape // tc if ts is None else ts
        msg = (
            f'Axis {axis_tiling.axis} cannot be split into'
            f' {axis_tiling.tile_count} tiles.'
        )
        assert axis_tiling.tile_size * axis_tiling.tile_count == axis_shape, msg
      # add missing tiled axis
      axis_in_cfg = [ax.axis for ax in axes_cfg]
      for axis in axes:
        if axis not in axis_in_cfg:
          axes_cfg.append(
              AxisTiling(axis=axis, tile_count=shape[axis], tile_size=1)
          )

    f(new_cfg.lhs.contraction_axes, lhs_ca, lhs_shape)
    f(new_cfg.lhs.remaining_axes, lhs_ra, lhs_shape)
    f(new_cfg.rhs.contraction_axes, rhs_ca, rhs_shape)
    f(new_cfg.rhs.remaining_axes, rhs_ra, rhs_shape)
    return new_cfg


def interleave(ls, rs):
  assert len(ls) == len(rs)
  ret = []
  for l, r in zip(ls, rs):
    ret.append(l)
    ret.append(r)
  return ret


def zip_product(l, r):
  return map(lambda x, y: x * y, l, r)


def get_ra(rank, ca, ba) -> list[AxisIdx]:
  return list(a for a in range(rank) if a not in ca + ba)


def maybe_add_one(i, min_i):
  return i + int(i >= min_i)


@dataclasses.dataclass(frozen=False, slots=True)
class TilingState:
  """Structure for bookkeeping of AxisIdx while tiling."""

  untiled_shape: tuple[AxisSize, ...]

  # Below are axes indices, similar to dimension_numbers
  # E.g.
  #   assert xhs.shape == [10, 20, 30, ...],
  #   assert tile_map == {0:[0], 1:[1], 2:[2], ...}
  # After splitting axis 1 to 2 tiles, sized 10, we get
  #   assert xhs.shape == [10, 2, 10, 30, ...]
  #   assert tile_map == {0:[0], 1:[1,2], 2:[3], ...}
  tile_map: dict[AxisIdx | str, list[AxisIdx]] = utils.dataclass_field(dict)

  tiled_shape: list[AxisSize] = utils.dataclass_field(list)

  def __post_init__(self):
    for i in range(len(self.untiled_shape)):
      # No axis is split at all yet.
      self.tile_map[i] = [i]
    self.tiled_shape = list(self.untiled_shape)

  def tile_axis(self, at: AxisTiling):
    """Tiles (splits) one axis while maintaining all AxisIdx."""
    msg = "Can't tile as all tiling must be done before broadcast operations."
    assert len(self.get_broadcasted_tile_map_indexes()) == 0, msg
    tile_axis = self.tile_map[at.axis]
    assert len(tile_axis) == 1, "can't tile the same axis twice."
    tile_axis = tile_axis[0]

    msg = f'{self.tiled_shape[tile_axis]=}, {at.tile_size=}, {at.tile_count=}'
    assert self.tiled_shape[tile_axis] == at.tile_size * at.tile_count, msg
    self.tiled_shape[tile_axis] = at.tile_size
    self.tiled_shape.insert(tile_axis, at.tile_count)

    # Update tile_map
    for k in self.tile_map:
      self.tile_map[k] = [
          maybe_add_one(ai, tile_axis) for ai in self.tile_map[k]
      ]
    self.tile_map[at.axis] = [tile_axis, tile_axis + 1]

  def tile_axes(self, ats: Iterable[AxisTiling]):
    for at in ats:
      self.tile_axis(at)

  def _is_broadcasted_ax(self, tile_map_idx: AxisIdx | str) -> bool:
    return isinstance(tile_map_idx, str) and tile_map_idx.startswith(
        BROADCAST_PREFIX
    )

  def get_broadcasted_tile_map_indexes(self):
    """Returns the list of keys in `self.tile_map` associated with broadcasting axes."""
    return [ax for ax in self.tile_map if self._is_broadcasted_ax(ax)]

  def apply(self, x: jnp.ndarray) -> jnp.ndarray:
    num_bcast_axes = len(self.get_broadcasted_tile_map_indexes())
    tiled_x = x.reshape(self.tiled_shape[num_bcast_axes:])
    tiled_x = jnp.broadcast_to(tiled_x, self.tiled_shape)
    return tiled_x

  def broadcast_to_other(self, bcast_shape: tuple[AxisSize, ...]):
    """Adds new axes (bcast_shape) on AxisIdx=0."""
    for k in self.tile_map:
      self.tile_map[k] = [ai + len(bcast_shape) for ai in self.tile_map[k]]
    keys = []
    for i in range(len(bcast_shape)):
      k = f'{BROADCAST_PREFIX}{i}'
      keys.append(k)
      assert k not in self.tile_map
      self.tile_map[k] = [i]
    self.tiled_shape = list(bcast_shape) + self.tiled_shape
    return keys

  def axes_shape(self, axes: list[AxisIdx]) -> tuple[AxisSize, ...]:
    return tuple(map(lambda a: self.tiled_shape[a], axes))

  def to_tiled_axes_transposed(
      self,
      axes: Iterable[AxisIdx | str],
      tile_level: int,
  ) -> tuple[list[AxisIdx], ...]:
    # pylint: disable=g-doc-args,g-doc-return-or-yield
    """The given 'axes' parameter defines axes in untiled represented Array.

    Function returns the corresponding AxisIdxs in self.x which is tiled. So for
    each give element of axes we can have multiple tiled axes. Function requires
    that the tile granularity is the same for all axes and equal to the given
    tile_level.

    tile_level = 1 for untiled axis
    tile_level = 2 for normally (one axis split into two) tiled.
    """
    lsts = [self.tile_map[ai] for ai in axes]
    msg = (
        'all requested axes should have the same tile level, but we have:'
        f' {lsts=} for {axes=}'
    )
    assert all(map(lambda tiled_ais: len(tiled_ais) == tile_level, lsts)), msg
    if lsts == []:
      return ([],) * tile_level
    return tuple(map(list, zip(*lsts)))


def print_dimension_numbers(dimension_numbers, lhs, rhs, label) -> None:
  """Prints dimension numbers before and/or after tiling.

  Args:
    dimension_numbers: It contains the left and right hand side contraction axes
      and batch axes.
    lhs: left hand side tensor.
    rhs: right hand side tensor.
    label: A string tag for logging info.
  """
  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
  lhs_ra = get_ra(lhs.ndim, lhs_ca, lhs_ba)
  rhs_ra = get_ra(rhs.ndim, rhs_ca, rhs_ba)
  logging.vlog(1, label)
  logging.vlog(1, f' lhs.shape={lhs.shape}')
  logging.vlog(1, f' rhs.shape={rhs.shape}')
  logging.vlog(1, f'lhs_ca={lhs_ca}')
  logging.vlog(1, f'rhs_ca={rhs_ca}')
  logging.vlog(1, f'lhs_ba={lhs_ba}')
  logging.vlog(1, f'rhs_ba={rhs_ba}')
  logging.vlog(1, f'lhs_ra={lhs_ra}')
  logging.vlog(1, f'rhs_ra={rhs_ra}')


def generate_tiling_states_for_dot_general(
    cfg: Cfg,
    lhs,
    rhs,
    dimension_numbers: jax.lax.DotDimensionNumbers,
) -> tuple[TilingState, TilingState]:
  """Do tiling for `lhs` and `rhs` and returns the intermediate tiling states.

  Args:
    cfg: Configuration to define how tiling should be done
    lhs: The left operand of dot general
    rhs: The right operand of dot general
    dimension_numbers: dimension numbers to indicate batch dimensions and
      contracting dimensions

  Returns:
    xlhs: Tiling state of the left operand `lhs` after the tiling
    xrhs: Tiling state of the left operand `rhs` after the tiling
  """
  logging.vlog(1, 'Tiling config cfg: %s', cfg)
  print_dimension_numbers(dimension_numbers, lhs, rhs, label='before tiling')

  # Config pre-processing and verification

  cfg = copy.deepcopy(cfg)
  cfg = cfg.complete_missing(lhs.shape, rhs.shape, dimension_numbers)
  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
  lhs_ra = get_ra(lhs.ndim, lhs_ca, lhs_ba)
  rhs_ra = get_ra(rhs.ndim, rhs_ca, rhs_ba)
  g_msg = (
      'Before tiling: \n'
      f'lhs: {lhs.shape=}, {lhs_ca=}, {lhs_ba=}, {lhs_ra=} \n'
      f'rhs: {rhs.shape=}, {rhs_ca=}, {rhs_ba=}, {rhs_ra=} \n'
      f'tiling cfg: {pprint.pformat(cfg)} \n'
  )

  # First tile_axis CA. CA tile_count axes will be first in xxhs.ba_tile
  assert len(cfg.lhs.contraction_axes) == len(cfg.rhs.contraction_axes), g_msg
  for cfg_lhs_ca, cfg_rhs_ca in zip(
      cfg.lhs.contraction_axes, cfg.rhs.contraction_axes
  ):
    msg = (
        'Contraction axis tile counts should be the same, but found lhs axis'
        f' {cfg_lhs_ca.axis} has a tile count of {cfg_lhs_ca.tile_count}, and'
        f' rhs axis {cfg_rhs_ca.axis} has a tile count of'
        f' {cfg_rhs_ca.tile_count}'
    )
    assert cfg_lhs_ca.tile_count == cfg_rhs_ca.tile_count, msg

  # Tiling

  xlhs = TilingState(untiled_shape=lhs.shape)
  xlhs.tile_axes(cfg.lhs.contraction_axes + cfg.lhs.remaining_axes)
  xrhs = TilingState(untiled_shape=rhs.shape)
  xrhs.tile_axes(cfg.rhs.contraction_axes + cfg.rhs.remaining_axes)

  # DotGeneral reshapeing

  _, xlhs_ra = xlhs.to_tiled_axes_transposed(lhs_ra, 2)
  xrhs.broadcast_to_other(xlhs.axes_shape(xlhs_ra))

  _, xrhs_ra = xrhs.to_tiled_axes_transposed(rhs_ra, 2)
  xlhs.broadcast_to_other(xrhs.axes_shape(xrhs_ra))

  return xlhs, xrhs


def tiled_dot_general_with_tiling_states(
    lhs: jnp.ndarray,
    xlhs: TilingState,
    rhs: jnp.ndarray,
    xrhs: TilingState,
    untiled_dimension_numbers: jax.lax.DotDimensionNumbers,
    precision=None,
    preferred_element_type=None,
    dot_general=jax.lax.dot_general,
):
  """local dot_general with tiling states."""
  xlhs_x = xlhs.apply(lhs)
  xrhs_x = xrhs.apply(rhs)
  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = untiled_dimension_numbers
  lhs_ra = get_ra(lhs.ndim, lhs_ca, lhs_ba)
  rhs_ra = get_ra(rhs.ndim, rhs_ca, rhs_ba)
  xlhs_ca_tile, xlhs_ca = xlhs.to_tiled_axes_transposed(lhs_ca, 2)
  xlhs_ra_tile, xlhs_ra = xlhs.to_tiled_axes_transposed(lhs_ra, 2)
  xrhs_ca_tile, xrhs_ca = xrhs.to_tiled_axes_transposed(rhs_ca, 2)
  xrhs_ra_tile, xrhs_ra = xrhs.to_tiled_axes_transposed(rhs_ra, 2)
  xlhs_bcast = xlhs.get_broadcasted_tile_map_indexes()
  xrhs_bcast = xrhs.get_broadcasted_tile_map_indexes()
  (xlhs_ra_other,) = xlhs.to_tiled_axes_transposed(xlhs_bcast, 1)
  (xrhs_ra_other,) = xrhs.to_tiled_axes_transposed(xrhs_bcast, 1)
  (xlhs_ba,) = xlhs.to_tiled_axes_transposed(lhs_ba, 1)
  (xrhs_ba,) = xrhs.to_tiled_axes_transposed(rhs_ba, 1)

  tiled_ca = (xlhs_ca, xrhs_ca)
  tiled_ba = (
      xlhs_ca_tile + xlhs_ba + xlhs_ra + xlhs_ra_other,
      xrhs_ca_tile + xrhs_ba + xrhs_ra_other + xrhs_ra,
  )
  tiled_dimension_numbers = (tiled_ca, tiled_ba)

  tiled_lhs_ra = get_ra(lhs.ndim, tiled_ca[0], tiled_ba[0])
  tiled_rhs_ra = get_ra(rhs.ndim, tiled_ca[1], tiled_ba[1])
  g_msg = (
      'After tiling: \n'
      f' lhs: lhs.shape={xlhs_x.shape}, lhs_ca={tiled_ca[0]},'
      f' lhs_ba={tiled_ba[0]}, lhs_ra={tiled_lhs_ra} \n'
      f' rhs: rhs.shape={xrhs_x.shape}, rhs_ca={tiled_ca[1]},'
      f' rhs_ba={tiled_ba[1]}, rhs_ra={tiled_rhs_ra} \n'
  )
  for axis in tiled_ca[0] + tiled_ba[0]:
    assert axis >= 0 and axis < xlhs_x.ndim, g_msg
  for axis in tiled_ca[1] + tiled_ba[1]:
    assert axis >= 0 and axis < xrhs_x.ndim, g_msg

  out = dot_general(
      xlhs_x, xrhs_x, tiled_dimension_numbers, precision, preferred_element_type
  )

  print_dimension_numbers(
      tiled_dimension_numbers, xlhs_x, xrhs_x, label='after tiling'
  )

  # Some assertions
  assert xlhs.axes_shape(xlhs_ca_tile) == xrhs.axes_shape(xrhs_ca_tile), g_msg
  ca_tile_sh = xlhs.axes_shape(xlhs_ca_tile)

  assert xlhs.axes_shape(xlhs_ba) == xrhs.axes_shape(xrhs_ba), g_msg
  ba_sh = xlhs.axes_shape(xlhs_ba)

  assert xlhs.axes_shape(xlhs_ra) == xrhs.axes_shape(xrhs_ra_other), g_msg
  lhs_ra_sh = xlhs.axes_shape(xlhs_ra)

  assert xlhs.axes_shape(xlhs_ra_other) == xrhs.axes_shape(xrhs_ra), g_msg
  rhs_ra_sh = xlhs.axes_shape(xlhs_ra_other)

  lhs_ra_tile_sh = xlhs.axes_shape(xlhs_ra_tile)
  rhs_ra_tile_sh = xrhs.axes_shape(xrhs_ra_tile)

  g_msg += f'Tiled dg {out.shape=} \n'
  assert (
      out.shape
      == ca_tile_sh
      + ba_sh
      + lhs_ra_sh
      + rhs_ra_sh
      + lhs_ra_tile_sh
      + rhs_ra_tile_sh
  ), g_msg

  # Sum over ca_tile now.
  # all_ba() returns ca_tile as the first axes in ba.
  assert len(xlhs_ca_tile) == len(xrhs_ca_tile)
  out = out.sum(axis=range(len(xlhs_ca_tile)))

  g_msg += f'After sum over tiles {out.shape=} \n'
  assert (
      out.shape
      == ba_sh + lhs_ra_sh + rhs_ra_sh + lhs_ra_tile_sh + rhs_ra_tile_sh
  ), g_msg

  # Transpose tile and tile size together
  # Axis tracking is obsolete here. Only the length is the same.
  start = 0
  end_ba = len(ba_sh)
  end_lhs_ra_tile = end_ba + len(lhs_ra_tile_sh)
  end_rhs_ra_tile = end_lhs_ra_tile + len(rhs_ra_tile_sh)
  end_lhs_ra = end_rhs_ra_tile + len(lhs_ra_sh)
  end_rhs_ra = end_lhs_ra + len(rhs_ra_sh)

  from_to = lambda end1, end2: list(range(end1, end2))

  new_ba = from_to(start, end_ba)
  new_lhs_ra_tile = from_to(end_ba, end_lhs_ra_tile)
  new_rhs_ra_tile = from_to(end_lhs_ra_tile, end_rhs_ra_tile)
  new_lhs_ra = from_to(end_rhs_ra_tile, end_lhs_ra)
  new_rhs_ra = from_to(end_lhs_ra, end_rhs_ra)

  out = out.transpose(
      new_ba
      + interleave(new_lhs_ra_tile, new_lhs_ra)
      + interleave(new_rhs_ra_tile, new_rhs_ra)
  )

  lhs_ra_sh_interleaved = tuple(interleave(lhs_ra_sh, lhs_ra_tile_sh))
  rhs_ra_sh_interleaved = tuple(interleave(rhs_ra_sh, rhs_ra_tile_sh))
  lhs_ra_sh_flattened = tuple(zip_product(lhs_ra_sh, lhs_ra_tile_sh))
  rhs_ra_sh_flattened = tuple(zip_product(rhs_ra_sh, rhs_ra_tile_sh))

  g_msg += f'After transpose {out.shape=} \n'
  assert (
      out.shape == ba_sh + lhs_ra_sh_interleaved + rhs_ra_sh_interleaved
  ), g_msg
  out = out.reshape(ba_sh + lhs_ra_sh_flattened + rhs_ra_sh_flattened)

  return out


def tiled_dot_general(
    cfg: Cfg,
    lhs,
    rhs,
    dimension_numbers,
    precision=None,
    preferred_element_type=None,
    dot_general=jax.lax.dot_general,
):
  """local dot_general."""
  xlhs, xrhs = generate_tiling_states_for_dot_general(
      cfg, lhs, rhs, dimension_numbers
  )
  return tiled_dot_general_with_tiling_states(
      lhs,
      xlhs,
      rhs,
      xrhs,
      dimension_numbers,
      precision,
      preferred_element_type,
      dot_general,
  )
