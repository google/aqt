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
from typing import Literal
import jax
import jax.numpy as jnp
import numpy as np


AxisIdx = int
AxisSize = int


@dataclasses.dataclass(frozen=False, slots=True)
class AxisTiling:
  axis: AxisIdx
  tile_count: AxisSize
  tile_size: AxisSize


@dataclasses.dataclass(frozen=False, slots=True)
class TensorTiling:
  contraction_axes: list[AxisTiling]
  remaining_axes: list[AxisTiling]
  # There is no point in tiling dot_general's batch axes


@dataclasses.dataclass(frozen=False, slots=True)
class Cfg:
  # Sequence of (lhs, rhs) configurations,
  # has to be the same lenght as both ca in dimension_numbers[0]
  lhs: TensorTiling
  rhs: TensorTiling


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


def sort_ra_cfg(ra_tiling: list[AxisTiling]) -> list[AxisTiling]:
  ra_axes = [a.axis for a in ra_tiling]
  sorted_idx = np.argsort(ra_axes).tolist()
  return [ra_tiling[i] for i in sorted_idx]


def empty_list():
  return dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=False, slots=True)
class _Xhs:
  """Structure for bookkeeping of AxisIdx while tiling."""

  x: jnp.ndarray

  # Below are axes indices, similar to dimension_numbers
  # E.g.
  #   xhs,shape == [10, 20, 30, ...] ca=[0,1,2], ca_tile=[]
  #   After splitting axis 1 to 2 tiles, sized 10, we get
  #   xhs,shape == [10, 2, 10, 30, ...], ca=[0,2,3], ca_tile=[1]
  #   and other axes indices (ra, ra_tile, ba) also can be updated by +1.
  ca: list[AxisIdx]
  ra: list[AxisIdx]

  ca_tile: list[AxisIdx] = empty_list()  # to be summed after DG
  ra_tile: list[AxisIdx] = empty_list()  # to be reshaped after DG
  ra_tile_other: list[AxisIdx] = empty_list()  # to be reshaped after DG
  # There is no point in tiling dot_general's batch axes
  ba: list[AxisIdx] = dataclasses.field(default_factory=list)

  # axes waiting to be slitted
  # tensor_tiling: TensorTiling # TODO(lew): use this
  ca_to_be_tiled: list[AxisTiling] = empty_list()
  ra_to_be_tiled: list[AxisTiling] = empty_list()

  def tile_axis(self, at: AxisTiling, tile_axis_name: Literal['ca', 'ra']):
    """Tiles (splits) one axis while maintaining all AxisIdx."""
    shape = list(self.x.shape)
    msg = f'{shape[at.axis]=}, {at.tile_size=}, {at.tile_count=}'
    assert shape[at.axis] == at.tile_size * at.tile_count, msg
    shape[at.axis] = at.tile_size
    shape.insert(at.axis, at.tile_count)
    self.x = self.x.reshape(shape)

    def update(axes):
      for i in range(len(axes)):
        axes[i] += int(axes[i] >= at.axis)

    def update_tiling(axes):
      for i in range(len(axes)):
        axes[i].axis += int(axes[i].axis >= at.axis)

    update(self.ca)
    update(self.ca_tile)
    update(self.ra)
    update(self.ra_tile)
    update(self.ra_tile_other)
    update(self.ba)

    update_tiling(self.ca_to_be_tiled)
    update_tiling(self.ra_to_be_tiled)
    # Append tile axis index at the beginning. This has to be after the update
    match tile_axis_name:
      case 'ca':
        self.ca_tile.append(at.axis)
      case 'ra':
        self.ra_tile.append(at.axis)

  def broadcast_to_other(self, bcast_shape: tuple[AxisSize, ...]):
    """Adds new axes (bcast_shape) on AxisIdx=0."""
    assert self.ra_tile_other == list(), 'ra_tile_other already set'
    self.ra_tile_other = list(range(len(bcast_shape)))
    self.x = jnp.broadcast_to(self.x, bcast_shape + self.x.shape)

    def update(axes):
      for i in range(len(axes)):
        axes[i] += len(bcast_shape)

    update(self.ca)
    update(self.ca_tile)
    update(self.ra)
    update(self.ra_tile)
    update(self.ba)

  def axes_shape(self, axes: list[AxisIdx]) -> tuple[AxisSize, ...]:
    return tuple(map(lambda a: self.x.shape[a], axes))


def tile(cfg, lhs, rhs, dimension_numbers):
  """Tile (split) contraction and remaining axes of lhs and rhs to dg."""
  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers

  xlhs = _Xhs(
      x=lhs,
      ca=list(lhs_ca),
      ra=get_ra(lhs.ndim, lhs_ca, lhs_ba),
      ba=list(lhs_ba),
      ca_to_be_tiled=cfg.lhs.contraction_axes,
      ra_to_be_tiled=sort_ra_cfg(cfg.lhs.remaining_axes),
  )
  xrhs = _Xhs(
      x=rhs,
      ca=list(rhs_ca),
      ra=get_ra(rhs.ndim, rhs_ca, rhs_ba),
      ba=list(rhs_ba),
      ca_to_be_tiled=cfg.rhs.contraction_axes,
      ra_to_be_tiled=sort_ra_cfg(cfg.rhs.remaining_axes),
  )

  # First tile_axis CA. CA tile_count axes will be first in xxhs.ba_tile
  # TODO(yichizh): move the assertion to cfg translator
  assert len(xlhs.ca_to_be_tiled) == len(xrhs.ca_to_be_tiled)
  while len(xlhs.ca_to_be_tiled) > 0:
    cfg_lhs_ca = xlhs.ca_to_be_tiled.pop(0)
    cfg_rhs_ca = xrhs.ca_to_be_tiled.pop(0)
    xlhs.tile_axis(cfg_lhs_ca, 'ca')
    xrhs.tile_axis(cfg_rhs_ca, 'ca')

  while len(xlhs.ra_to_be_tiled) > 0:
    cfg_lhs_ra = xlhs.ra_to_be_tiled.pop(0)
    xlhs.tile_axis(cfg_lhs_ra, 'ra')
  while len(xrhs.ra_to_be_tiled) > 0:
    cfg_rhs_ra = xrhs.ra_to_be_tiled.pop(0)
    xrhs.tile_axis(cfg_rhs_ra, 'ra')

  xlhs.broadcast_to_other(xrhs.axes_shape(xrhs.ra_tile))
  xrhs.broadcast_to_other(xlhs.axes_shape(xlhs.ra_tile))

  tiled_ca = (xlhs.ca, xrhs.ca)
  tiled_ba = (
      xlhs.ca_tile + xlhs.ba + xlhs.ra_tile + xlhs.ra_tile_other,
      xrhs.ca_tile + xrhs.ba + xrhs.ra_tile_other + xrhs.ra_tile,
  )
  new_dimension_numbers = (tiled_ca, tiled_ba)
  return new_dimension_numbers, xlhs, xrhs


def tiled_dot_general(
    cfg: Cfg,
    lhs,
    rhs,
    dimension_numbers,
    precision=None,
    preferred_element_type=None,
):
  """local dot_general."""

  cfg = copy.deepcopy(cfg)
  new_dimension_numbers, xlhs, xrhs = tile(cfg, lhs, rhs, dimension_numbers)

  # Core dot general call that can be replaced by dg injection
  out = jax.lax.dot_general(
      xlhs.x, xrhs.x, new_dimension_numbers, precision, preferred_element_type
  )

  # Some assertions
  assert xlhs.axes_shape(xlhs.ca_tile) == xrhs.axes_shape(xrhs.ca_tile)
  ca_tile_sh = xlhs.axes_shape(xlhs.ca_tile)

  assert xlhs.axes_shape(xlhs.ba) == xrhs.axes_shape(xrhs.ba)
  ba_sh = xlhs.axes_shape(xlhs.ba)

  assert xlhs.axes_shape(xlhs.ra_tile) == xrhs.axes_shape(xrhs.ra_tile_other)
  lhs_ra_tile_sh = xlhs.axes_shape(xlhs.ra_tile)

  assert xlhs.axes_shape(xlhs.ra_tile_other) == xrhs.axes_shape(xrhs.ra_tile)
  rhs_ra_tile_sh = xlhs.axes_shape(xlhs.ra_tile_other)

  lhs_ra_sh = xlhs.axes_shape(xlhs.ra)
  rhs_ra_sh = xrhs.axes_shape(xrhs.ra)

  assert (
      out.shape
      == ca_tile_sh
      + ba_sh
      + lhs_ra_tile_sh
      + rhs_ra_tile_sh
      + lhs_ra_sh
      + rhs_ra_sh
  )

  # Sum over ca_tile now.
  # all_ba() returs ca_tile as the first axes in ba.
  assert len(xlhs.ca_tile) == len(xrhs.ca_tile)
  out = out.sum(axis=range(len(xlhs.ca_tile)))

  assert (
      out.shape
      == ba_sh + lhs_ra_tile_sh + rhs_ra_tile_sh + lhs_ra_sh + rhs_ra_sh
  )

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

  lhs_ra_sh_interleaved = tuple(interleave(lhs_ra_tile_sh, lhs_ra_sh))
  rhs_ra_sh_interleaved = tuple(interleave(rhs_ra_tile_sh, rhs_ra_sh))
  lhs_ra_sh_flattened = tuple(zip_product(lhs_ra_tile_sh, lhs_ra_sh))
  rhs_ra_sh_flattened = tuple(zip_product(rhs_ra_tile_sh, rhs_ra_sh))

  assert out.shape == ba_sh + lhs_ra_sh_interleaved + rhs_ra_sh_interleaved
  out = out.reshape(ba_sh + lhs_ra_sh_flattened + rhs_ra_sh_flattened)

  return out
