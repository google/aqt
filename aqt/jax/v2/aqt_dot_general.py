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
"""Quantized dot_general."""

# Lingo in this file:
#
# - lhs(rhs) - left(right) hand side of a binary operation
# - ca - contraction axes
# - ba - batch axes
# - ra - remaining axes

# pylint: disable=g-explicit-bool-comparison
# pylint: disable=g-explicit-length-test

import functools
from typing import Callable, Optional, Union

from aqt.jax.v2 import aqt_tensor
from aqt.jax.v2 import config
# TODO(yichizh): The following import is temporary for not breaking dependencies
# Fix imports in other packages and delete it.
from aqt.jax.v2.config import Context  # pylint: disable=g-importing-member, unused-import
import flax.struct
import jax
from jax import lax
import jax.numpy as jnp
import numpy as onp


@flax.struct.dataclass
class MultiTensor:
  x: jnp.ndarray
  qx: aqt_tensor.QTensor


@flax.struct.dataclass
class TensorRes:
  """All the things we pass from the forward pass to the backward pass."""
  mt: MultiTensor
  quant_grad: Union[Callable[[jnp.ndarray], tuple[jnp.ndarray]], None]


@flax.struct.dataclass
class DotGeneralRes:
  lhs: TensorRes
  rhs: TensorRes


def _scale_trans(x, ca, ba):
  """Transposes x to output dimension order."""
  ca = list(ca)
  ba = list(ba)
  for i in ca:
    assert x.shape[i] == 1
  ra = list(i for i in range(len(x.shape)) if i not in ba + ca)
  x = jnp.transpose(x, ba + ra + ca)
  # TODO(lew): x = jnp.squeeze(x, axis=range(len(ba+ra): len(x.shape))
  shape_ba = x.shape[: len(ba)]
  shape_ra = x.shape[len(ba) : len(x.shape) - len(ca)]
  # Will need to add additional axes (size 1) for the other shape_ra
  x = x.reshape(shape_ba + shape_ra)
  return x


def _lhs_scale_transpose(lhs_scale, dimension_numbers, lhs_shape, rhs_shape):
  """Transposes lhs_scale to output dimension order."""
  if lhs_scale is None:
    return None
  # The axis order in out is as follows: batch, lhs_ra, rhs_ra
  # - batch axes order is uniquely determined by either lhs_ba or rhs_ba
  # - contraction axes ca disappear from the output
  # - order of the remaining axes (ra) is preserved.
  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
  qlhs_scale_t = _scale_trans(lhs_scale, lhs_ca, lhs_ba)
  # inserting dummy axes for rhs_ra
  assert len(qlhs_scale_t.shape) == len(lhs_shape) - len(lhs_ca)
  start = len(qlhs_scale_t.shape)
  end = len(rhs_shape) - len(rhs_ca) - len(rhs_ba) + start
  lhs_dummy_axes = range(start, end)
  qlhs_scale_t = jnp.expand_dims(qlhs_scale_t, axis=lhs_dummy_axes)
  return qlhs_scale_t


def _rhs_scale_transpose(rhs_scale, dimension_numbers, lhs_shape, rhs_shape):
  if rhs_scale is None:
    return None
  del rhs_shape
  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
  qrhs_scale_t = _scale_trans(rhs_scale, rhs_ca, rhs_ba)
  start = len(rhs_ba)
  end = len(lhs_shape) - len(lhs_ca) - len(lhs_ba) + start
  rhs_dummy_axes = range(start, end)
  qrhs_scale_t = jnp.expand_dims(qrhs_scale_t, axis=rhs_dummy_axes)
  return qrhs_scale_t


def _maybe_mul(x, scale):
  if scale is None:
    return x
  return x * scale


def _make_dot_general_raw(cfg: config.DotGeneralRaw):
  """Makes quantized lax.dot_general replacement."""

  msg = 'Custom calib_shared_axes not implemented for local AQT.'
  assert cfg.lhs.calib_shared_axes is None, msg
  assert cfg.rhs.calib_shared_axes is None, msg

  def dot_general_raw(
      lhs: jnp.ndarray,
      rhs: Union[jnp.ndarray, MultiTensor],
      # xhs_qt are used in serving.
      lhs_qt: Optional[aqt_tensor.QTensor],
      rhs_qt: Optional[aqt_tensor.QTensor],
      dimension_numbers: jax.lax.DotDimensionNumbers,
  ):
    """Creates a dot_general function without custom gradient."""
    (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
    # We need to copy because we modify cfg to populate some defaults.

    # TODO(lew):
    #  - Use qx.value with the int type.
    #  - Handle qx.value with the int type in an optimized way.
    #  - Add a "FQ" case we multiply qx.value*qx.value_scale (not transposed).
    #  - Can we carry untransposed scale and transpose here?
    if isinstance(rhs, MultiTensor):
      # We are in gradient code.
      fwd_quantized = rhs.qx.scale_t is not None
      expect_fwd_quantized = cfg.rhs.use_fwd_quant is not None
      msg = (
          'Misconfiguration: use_fwd_quant=True, but there is no fwd'
          ' quantization (but rhs.qx is None).'
      )
      assert fwd_quantized == expect_fwd_quantized, msg
      if cfg.rhs.use_fwd_quant:
        assert rhs.qx.scale_t is not None, msg
        lhs = lhs * rhs.qx.scale_t
        rhs = rhs.qx.qvalue
      else:
        rhs = rhs.x
    else:
      assert cfg.rhs.use_fwd_quant is None, 'cannot set use_fwd_quant in fwd'

    if cfg.local_aqt is not None:

      def factor_reshape(x, ca, ba):
        factor = cfg.local_aqt.contraction_axis_shard_count
        assert factor is not None
        if len(ca) == 0:
          return x, ca, ba
        shape = list(x.shape)
        ax = ca[0]
        orig_size = shape[ax]
        assert orig_size % factor == 0
        shape[ax] = factor
        shape.insert(ax + 1, orig_size // factor)
        new_ca = [(b + int(b >= ax)) for b in ca]
        assert new_ca[0] == ax + 1
        new_ba = [ax] + [(b + int(b > ax)) for b in ba]
        return x.reshape(shape), new_ca, new_ba

      lhs, lhs_ca, lhs_ba = factor_reshape(lhs, lhs_ca, lhs_ba)
      rhs, rhs_ca, rhs_ba = factor_reshape(rhs, rhs_ca, rhs_ba)

      dimension_numbers = (lhs_ca, rhs_ca), (lhs_ba, rhs_ba)

    assert isinstance(rhs, jnp.ndarray)

    # TODO(lew): Have a function to handle lhs and rhs uniformly.
    if lhs_qt is not None:
      lhs_quant_grad = 'Poison. Not needed in serving'
    else:
      transpose = functools.partial(
          _lhs_scale_transpose,
          dimension_numbers=dimension_numbers,
          lhs_shape=lhs.shape,
          rhs_shape=rhs.shape,
      )

      lhs_qt, lhs_quant_grad = aqt_tensor.quant(
          lhs, cfg=cfg.lhs, calibration_axes=lhs_ca, transpose_fn=transpose
      )

    lhs_mt = MultiTensor(x=lhs, qx=lhs_qt)
    lhs_res = TensorRes(mt=lhs_mt, quant_grad=lhs_quant_grad)

    if rhs_qt is not None:
      rhs_quant_grad = 'Poison. Not needed in serving'
    else:
      transpose = functools.partial(
          _rhs_scale_transpose,
          dimension_numbers=dimension_numbers,
          lhs_shape=lhs.shape,
          rhs_shape=rhs.shape,
      )
      rhs_qt, rhs_quant_grad = aqt_tensor.quant(
          rhs, cfg=cfg.rhs, calibration_axes=rhs_ca, transpose_fn=transpose
      )
    rhs_mt = MultiTensor(x=rhs, qx=rhs_qt)
    rhs_res = TensorRes(mt=rhs_mt, quant_grad=rhs_quant_grad)

    # TODO(lew): mt.x above should be clipped for clipping calibrations

    # TODO(yichizh): the same code is applied to lhs and rhs.
    # Should make a function of it that includes preprocess as well.
    lhs_cast_dtype = cfg.lhs.numerics.get_dtype()
    rhs_cast_dtype = cfg.rhs.numerics.get_dtype()
    msg = "Can't cast dtype in fake_quant mode."
    if cfg.lhs.use_fake_quant:
      # TODO(yichizh): replace rounding in numerics with casting to dtype.
      # So fake quant becomes casting to dtype first, then casting to bfloat.
      # This is because FP8 numerics relies on this cast to do the rounding.
      assert lhs_cast_dtype is None, msg
      lhs_qin = lhs_qt.dequant()
    else:
      lhs_qin = lhs_qt.qvalue
      if lhs_cast_dtype is not None:
        lhs_qin = lhs_qin.astype(lhs_cast_dtype)
    if cfg.rhs.use_fake_quant:
      assert rhs_cast_dtype is None, msg
      rhs_qin = rhs_qt.dequant()
    else:
      rhs_qin = rhs_qt.qvalue
      if rhs_cast_dtype is not None:
        rhs_qin = rhs_qin.astype(rhs_cast_dtype)

    dtype_ms = (
        f'Found {cfg.dg_accumulator_dtype=}, {lhs_cast_dtype=} and'
        f' {rhs_cast_dtype=}. Dot general accumulator dtype can only be'
        ' jnp.int32 when both inputs are int8. Otherwise it is recommended to'
        ' be None to let lax.dot_general automatically decide it.'
    )
    if cfg.dg_accumulator_dtype == jnp.int32:
      assert lhs_cast_dtype == jnp.int8 and rhs_cast_dtype == jnp.int8, dtype_ms
    out = lax.dot_general(
        lhs_qin,
        rhs_qin,
        dimension_numbers=dimension_numbers,
        preferred_element_type=cfg.dg_accumulator_dtype,
        precision=lax.Precision.DEFAULT,
    ).astype(jnp.promote_types(lhs, rhs))
    # TODO(lew): Do we have a correct precision above?
    #   Relevant: https://github.com/google/jax/issues/14022

    if not cfg.lhs.use_fake_quant:
      out = _maybe_mul(out, lhs_qt.scale_t)
    if not cfg.rhs.use_fake_quant:
      out = _maybe_mul(out, rhs_qt.scale_t)

    res = DotGeneralRes(
        lhs=lhs_res,
        rhs=rhs_res,
    )
    if cfg.local_aqt is not None:
      assert len(lhs_ca) == len(rhs_ca)
      if len(lhs_ca) > 0:
        out = jnp.sum(out, axis=0)
      # We are not supporting local AQT in fwd pass, so no res needed.
      res = None
    return out, res

  return dot_general_raw


def _dot_general_raw_attach_gradient(
    fwd_dot_general_raw,
    dlhs_dot_general_raw,
    drhs_dot_general_raw,
):
  """Makes quantized lax.dot_general replacement with attached gradients."""

  def make_fwd(return_residual):

    def fwd(
        lhs,
        rhs,
        lhs_qt,
        rhs_qt,
        dimension_numbers,
    ):
      assert (
          lhs.dtype == rhs.dtype
      ), f'Unmatched lhs and rhs dtype: {lhs.dtype} vs {rhs.dtype}'
      with jax.named_scope('aqt_fwd'):
        ret, res = fwd_dot_general_raw(
            lhs,
            rhs,
            lhs_qt,
            rhs_qt,
            dimension_numbers,
        )
        ret = ret.astype(lhs.dtype)
      # We return these values to allow for materialization.
      qret = (res.lhs.mt.qx, res.rhs.mt.qx)
      if return_residual:
        return ((ret, qret), res)
      else:
        return (ret, qret)

    return fwd

  def vjp_bwd(
      fwd_dimension_numbers,
      res: DotGeneralRes,
      g,
  ):
    # g[1] is gradient with respect to qret which we are ignoring.
    g = g[0]
    def ranges_like(*xs):
      start = 0
      for x in xs:
        yield tuple(range(start, start + len(x)))
        start += len(x)

    def grad_dot_general(
        y_res: TensorRes,
        quant_grad,
        dot_general,
        y_is_lhs,
    ):
      y_ndim = y_res.mt.x.ndim

      (x_ca, y_ca), (x_ba, y_ba) = fwd_dimension_numbers
      if y_is_lhs:
        (y_ca, x_ca) = (x_ca, y_ca)
        (y_ba, x_ba) = (x_ba, y_ba)
      g_ndim = g.ndim - y_ndim + len(x_ba) + 2 * len(x_ca)
      x_ra = tuple(i for i in range(g_ndim) if i not in x_ca and i not in x_ba)
      y_ra = tuple(i for i in range(y_ndim) if i not in y_ca and i not in y_ba)
      if y_is_lhs:
        g_ba, g_ca, _ = ranges_like(x_ba, y_ra, x_ra)
      else:
        g_ba, _, g_ca = ranges_like(x_ba, x_ra, y_ra)
      dims = ((g_ca, y_ra), (g_ba, y_ba))

      out, _ = dot_general(g, y_res.mt, None, None, dims)

      x_ca_sorted_by_y = tuple(onp.take(x_ca, onp.argsort(y_ca)))
      out_axes = tuple(onp.argsort(tuple(x_ba) + x_ra + x_ca_sorted_by_y))
      transposed_out = jax.lax.transpose(out, out_axes)
      if quant_grad is not None:
        transposed_out = quant_grad(transposed_out)[0]
      return transposed_out

    with jax.named_scope('aqt_dlhs'):
      dlhs = grad_dot_general(
          res.rhs,
          res.lhs.quant_grad,
          dlhs_dot_general_raw,
          False,
      )
    with jax.named_scope('aqt_drhs'):
      drhs = grad_dot_general(
          res.lhs,
          res.rhs.quant_grad,
          drhs_dot_general_raw,
          True,
      )
    # fwd_dimension_numbers are marked as nondiff_argnums instead of returning
    # None as grad to it. This is because it is a tuple of Python integers
    # that cannot be traced by Jax.
    return (dlhs, drhs, None, None)

  vjp = jax.custom_vjp(make_fwd(False), nondiff_argnums=(4,))
  vjp.defvjp(make_fwd(True), vjp_bwd)

  return vjp


def make_dot_general(cfg: Optional[config.DotGeneral]):
  """Makes quantized lax.dot_general replacement with attached gradients."""
  if cfg is None:
    return jax.lax.dot_general

  dg = _dot_general_raw_attach_gradient(
      fwd_dot_general_raw=_make_dot_general_raw(cfg.fwd),
      dlhs_dot_general_raw=_make_dot_general_raw(cfg.dlhs),
      drhs_dot_general_raw=_make_dot_general_raw(cfg.drhs),
  )

  def ret_dg(
      lhs,
      rhs,
      dimension_numbers,
      precision=None,
      preferred_element_type=None,
  ):
    del preferred_element_type
    assert (
        precision is None
    ), f'Precision {precision} requested together with quantization.'

    msg = 'AQT is not yet optimized to accept quantized types directly. '
    msg += f'lhs.dtype: {lhs.dtype}, rhs.dtype: {rhs.dtype}'
    assert lhs.dtype in [jnp.bfloat16, jnp.float32, jnp.float16], msg
    assert rhs.dtype in [jnp.bfloat16, jnp.float32, jnp.float16], msg

    def get_qt(fwd_tensor_cfg: config.Tensor):
      if fwd_tensor_cfg.preprocess is not None:
        return fwd_tensor_cfg.preprocess(None)
      return None

    def set_qt(fwd_tensor_cfg: config.Tensor, qt: aqt_tensor.QTensor) -> None:
      if fwd_tensor_cfg.preprocess is not None:
        dtype = fwd_tensor_cfg.numerics.get_dtype()
        qt_cast = aqt_tensor.QTensor(
            qt.qvalue.astype(dtype), scale=qt.scale, scale_t=qt.scale_t
        )
        fwd_tensor_cfg.preprocess(qt_cast)

    lhs_qt = get_qt(cfg.fwd.lhs)
    rhs_qt = get_qt(cfg.fwd.rhs)
    out, (out_lhs_qt, out_rhs_qt) = dg(
        lhs=lhs,
        rhs=rhs,
        lhs_qt=lhs_qt,
        rhs_qt=rhs_qt,
        dimension_numbers=dimension_numbers,
    )
    set_qt(cfg.fwd.lhs, out_lhs_qt)
    set_qt(cfg.fwd.rhs, out_rhs_qt)

    return out

  return ret_dg
