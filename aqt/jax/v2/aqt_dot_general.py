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
from typing import Any, Optional, Sequence, Union

from aqt.jax.v2 import aqt_tensor
from aqt.jax.v2 import config
from aqt.jax.v2 import utils
# TODO(yichizh): The following import is temporary for not breaking dependencies
# Fix imports in other packages and delete it.
import jax
from jax import lax
from jax._src.numpy import lax_numpy
import jax.numpy as jnp
import numpy as onp


@utils.flax_slots_dataclass
class MultiTensor:
  x: jnp.ndarray
  qx: aqt_tensor.QTensor


@utils.flax_slots_dataclass
class TensorRes:
  """All the things we pass from the forward pass to the backward pass."""
  mt: MultiTensor
  quant_grad: aqt_tensor.GradientFn


@utils.flax_slots_dataclass
class DotGeneralRes:
  lhs: TensorRes
  rhs: TensorRes


def einsum(eqn: str, lhs: jnp.ndarray, rhs: jnp.ndarray, dg=lax.dot_general):
  """A copy of jnp.einsum but without the default jit so as to be injectable."""
  operands, contractions = lax_numpy._default_poly_einsum_handler(  # pylint: disable=protected-access
      eqn, lhs, rhs, einsum_call=True, use_blas=True, optimize='optimal'
  )
  contractions = tuple((a, frozenset(b), c) for a, b, c, *_ in contractions)
  return jax.named_call(lax_numpy._einsum, name=eqn)(  # pylint: disable=protected-access
      operands,
      contractions,
      precision=None,
      preferred_element_type=None,
      _dot_general=dg,
  )


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


def _lhs_scale_transpose_to_output(
    lhs_scale, dimension_numbers, lhs_shape, rhs_shape
):
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


def _rhs_scale_transpose_to_output(
    rhs_scale, dimension_numbers, lhs_shape, rhs_shape
):
  """Transposes rhs_scale to output dimension order."""
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


def _scale_trans_for_other_input(
    x, my_ca, my_ba, other_ca, other_ba, other_rank
):
  """Transposes x to other inputs' dimension order."""
  my_ca = list(my_ca)
  my_ba = list(my_ba)
  other_ca = list(other_ca)
  other_ba = list(other_ba)

  # Match the rank.
  if len(x.shape) < other_rank:
    x = x.reshape(list(x.shape) + [1] * (other_rank - len(x.shape)))

  transpose_dim = [-1] * len(x.shape)
  my_axis_mapped = my_ca + my_ba
  other_axis_mapped = other_ca + other_ba
  my_ra = list(i for i in range(len(x.shape)) if i not in my_axis_mapped)
  for axis in my_ra:
    assert x.shape[axis] == 1
  for my_axis, other_axis in zip(my_axis_mapped, other_axis_mapped):
    transpose_dim[other_axis] = my_axis

  # Fill unrelated axis with remaining axis.
  ra_idx = 0
  for transpose_dim_idx, transpose_dim_value in enumerate(transpose_dim):
    if transpose_dim_value == -1:
      transpose_dim[transpose_dim_idx] = my_ra[ra_idx]
      ra_idx += 1
  assert ra_idx == len(my_ra)

  # Transpose.
  x = jnp.transpose(x, transpose_dim)

  # Remove redundant axis.
  if len(x.shape) > other_rank:
    for idx in range(len(x.shape), other_rank):
      assert x.shape[idx] == 1
    x = x.reshape(x.shape[:other_rank])

  return x


def _lhs_scale_transpose_for_rhs_input(lhs_scale, dimension_numbers, rhs_shape):
  """Transposes lhs_scale to rhs input dimension order."""
  if lhs_scale is None:
    return None

  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
  return _scale_trans_for_other_input(
      lhs_scale, lhs_ca, lhs_ba, rhs_ca, rhs_ba, len(rhs_shape)
  )


def _rhs_scale_transpose_for_lhs_input(rhs_scale, dimension_numbers, lhs_shape):
  """Transposes lhs_scale to rhs input dimension order."""
  if rhs_scale is None:
    return None

  (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
  return _scale_trans_for_other_input(
      rhs_scale, rhs_ca, rhs_ba, lhs_ca, lhs_ba, len(lhs_shape)
  )


def _get_scale_t(
    qt: aqt_tensor.QTensor,
    transpose_fn: Any,
    dimension_numbers: lax.DotDimensionNumbers,
    lhs_shape: Sequence[int],
    rhs_shape: Sequence[int],
) -> aqt_tensor.QTensor:
  list_scale_t = []
  for scale in qt.scale:
    scale_t = transpose_fn(scale, dimension_numbers, lhs_shape, rhs_shape)
    list_scale_t.append(scale_t)
  return qt.replace(scale_t=list_scale_t)


def _make_dot_general_raw(cfg: config.DotGeneralRaw):
  """Makes quantized lax.dot_general replacement."""

  msg = 'Custom calib_shared_axes not implemented for local AQT.'
  assert cfg.lhs.quantizer.calib_shared_axes is None, msg
  assert cfg.rhs.quantizer.calib_shared_axes is None, msg

  @jax.named_scope(cfg.jax_scope_name)
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
      fwd_quantized = rhs.qx.scale_t is not None and len(rhs.qx.scale_t) == 1  # pytype: disable=attribute-error
      expect_fwd_quantized = cfg.rhs.use_fwd_quant is not None
      msg = (
          'Misconfiguration: use_fwd_quant=True, but there is no fwd'
          ' quantization (but rhs.qx is None).'
      )
      assert fwd_quantized == expect_fwd_quantized, msg
      if cfg.rhs.use_fwd_quant:
        assert fwd_quantized, msg
        # TODO(lew): Investigate why _rhs_scale_transpose_for_lhs_input is not
        # needed here.
        lhs = lhs * rhs.qx.scale_t[0]  # pytype: disable=attribute-error
        rhs = rhs.qx.qvalue  # pytype: disable=attribute-error
      else:
        rhs = rhs.x  # pytype: disable=attribute-error
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

    def _compute_qtensor(
        inputs: jnp.ndarray,
        input_qtensor: aqt_tensor.QTensor,
        tensor_cfg: config.Tensor,
        calibration_axes: Sequence[int],
        transpose_fn: Any,
    ) -> tuple[aqt_tensor.QTensor, str | aqt_tensor.GradientFn]:
      """Compute qtensor from input or input_qtensor."""
      if input_qtensor is not None:
        quant_grad = 'Poison. Not needed in serving'
        output_qtensor = input_qtensor
      else:
        output_qtensor, quant_grad = tensor_cfg.quantizer.quant(
            inputs, calibration_axes=calibration_axes
        )
      if (
          output_qtensor.scale_t is None
          and tensor_cfg.dequant_mode != config.DequantMode.OTHER_INPUT
      ):
        msg = 'scale, scale_t cannot be both unknown'
        assert output_qtensor.scale is not None, msg
        get_scale_t = functools.partial(
            _get_scale_t,
            dimension_numbers=dimension_numbers,
            lhs_shape=lhs.shape,
            rhs_shape=rhs.shape,
        )
        output_qtensor = get_scale_t(output_qtensor, transpose_fn)
      return output_qtensor, quant_grad

    lhs_qt, lhs_quant_grad = _compute_qtensor(
        lhs, lhs_qt, cfg.lhs, lhs_ca, _lhs_scale_transpose_to_output
    )
    lhs_mt = MultiTensor(x=lhs, qx=lhs_qt)
    lhs_res = TensorRes(mt=lhs_mt, quant_grad=lhs_quant_grad)

    rhs_qt, rhs_quant_grad = _compute_qtensor(
        rhs, rhs_qt, cfg.rhs, rhs_ca, _rhs_scale_transpose_to_output
    )
    rhs_mt = MultiTensor(x=rhs, qx=rhs_qt)
    rhs_res = TensorRes(mt=rhs_mt, quant_grad=rhs_quant_grad)

    # TODO(lew): mt.x above should be clipped for clipping calibrations

    def _cast_dtype(
        input_qtensor: aqt_tensor.QTensor, tensor_cfg: config.Tensor
    ) -> jnp.ndarray:
      dtype = tensor_cfg.quantizer.numerics.get_dtype()
      msg = "Can't cast dtype in DequantMode.THIS_INPUT (fake quant) mode."
      if tensor_cfg.dequant_mode == config.DequantMode.THIS_INPUT:
        assert dtype is None, msg
        output = input_qtensor.dequant()
      else:
        # TODO(yichizh): replace rounding in numerics with casting to dtype.
        # So fake quant becomes casting to dtype first, then casting to bfloat.
        # This is because FP8 numerics relies on this cast to do the rounding.
        qvalue = input_qtensor.qvalue
        output = qvalue if dtype is None else qvalue.astype(dtype)
      return output

    lhs_qin = _cast_dtype(lhs_qt, cfg.lhs)
    rhs_qin = _cast_dtype(rhs_qt, cfg.rhs)

    lhs_cast_dtype = cfg.lhs.quantizer.numerics.get_dtype()
    rhs_cast_dtype = cfg.rhs.quantizer.numerics.get_dtype()
    dtype_ms = (
        f'Found {cfg.dg_accumulator_dtype=}, {lhs_cast_dtype=} and'
        f' {rhs_cast_dtype=}. Dot general accumulator dtype can only be'
        ' jnp.int32 when both inputs are int8. Otherwise it is recommended to'
        ' be None to let lax.dot_general automatically decide it.'
    )
    if cfg.dg_accumulator_dtype == jnp.int32:
      assert lhs_cast_dtype == jnp.int8 and rhs_cast_dtype == jnp.int8, dtype_ms

    if cfg.lhs.dequant_mode == config.DequantMode.OTHER_INPUT:
      assert rhs_qin.dtype in [jnp.float32, jnp.bfloat16, jnp.float64]
      for scale in lhs_qt.scale:
        rhs_qin = rhs_qin * _lhs_scale_transpose_for_rhs_input(
            scale, dimension_numbers, rhs.shape
        )

    if cfg.rhs.dequant_mode == config.DequantMode.OTHER_INPUT:
      assert lhs_qin.dtype in [jnp.float32, jnp.bfloat16, jnp.float64]
      for scale in rhs_qt.scale:
        lhs_qin = lhs_qin * _rhs_scale_transpose_for_lhs_input(
            scale, dimension_numbers, lhs.shape
        )

    out = lax.dot_general(
        lhs_qin,
        rhs_qin,
        dimension_numbers=dimension_numbers,
        preferred_element_type=cfg.dg_accumulator_dtype,
        precision=lax.Precision.DEFAULT,
    )
    # TODO(lew): Do we have a correct precision above?
    #   Relevant: https://github.com/google/jax/issues/14022

    out = aqt_tensor.QTensor(
        qvalue=out,
        scale=[],
        scale_t=None,
        dequant_dtype=jnp.promote_types(lhs, rhs),
    )
    assert out.scale is not None  # pytype help

    if cfg.lhs.dequant_mode == config.DequantMode.OUTPUT:
      out.scale.extend(lhs_qt.scale_t)
    if cfg.rhs.dequant_mode == config.DequantMode.OUTPUT:
      out.scale.extend(rhs_qt.scale_t)

    out = out.dequant()

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


# TODO(yichizh): inline this function. Make all child functions top-level.
def _dot_general_raw_attach_gradient():
  """Makes quantized lax.dot_general replacement with attached gradients."""

  def make_fwd(return_residual):

    # When defining a vjp, all traceable variables must be input arguments of
    # both the fwd and bwd function.
    # The cfg (DotGeneral) contains the key used for stochastic rounding,
    # which are traceable dynamic variables. It needs to be an input argument
    # to prevent the jax side effect.
    def fwd(
        lhs: jnp.ndarray,
        rhs: jnp.ndarray,
        lhs_qt: Optional[aqt_tensor.QTensor],
        rhs_qt: Optional[aqt_tensor.QTensor],
        dimension_numbers: lax.DotDimensionNumbers,
        cfg: config.DotGeneral,
    ):
      assert (
          lhs.dtype == rhs.dtype
      ), f'Unmatched lhs and rhs dtype: {lhs.dtype} vs {rhs.dtype}'
      fwd_dot_general_raw = _make_dot_general_raw(cfg.fwd)
      ret, res = fwd_dot_general_raw(
          lhs,
          rhs,
          lhs_qt,
          rhs_qt,
          dimension_numbers,
      )
      ret = ret.astype(lhs.dtype)
      # We return these values to allow for materialization.
      assert res is not None, 'res cannot be None in fwd pass.'
      qret = (res.lhs.mt.qx, res.rhs.mt.qx)
      if return_residual:
        return ((ret, qret), (res, cfg))
      else:
        return (ret, qret)

    return fwd

  def vjp_bwd(
      fwd_dimension_numbers: lax.DotDimensionNumbers,
      res: tuple[Optional[DotGeneralRes], config.DotGeneral],
      g,
  ):
    # g[1] is gradient with respect to qret which we are ignoring.
    dg_res, cfg = res
    msg = 'dg_res can only be None in 2nd derivative. It is not yet supported.'
    assert dg_res is not None, msg
    g = g[0]
    def ranges_like(*xs):
      start = 0
      for x in xs:
        yield tuple(range(start, start + len(x)))
        start += len(x)

    def grad_dot_general(
        y_res: TensorRes,
        quant_grad: aqt_tensor.GradientFn,
        cfg_raw: config.DotGeneralRaw,
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

      dot_general = _make_dot_general_raw(cfg_raw)
      out, _ = dot_general(g, y_res.mt, None, None, dims)

      x_ca_sorted_by_y = tuple(onp.take(x_ca, onp.argsort(y_ca)))
      out_axes = tuple(onp.argsort(tuple(x_ba) + x_ra + x_ca_sorted_by_y))
      transposed_out = jax.lax.transpose(out, out_axes)
      if quant_grad is not None:
        transposed_out = quant_grad(transposed_out)[0]
      return transposed_out

    dlhs = grad_dot_general(
        dg_res.rhs,
        dg_res.lhs.quant_grad,
        cfg.dlhs,
        False,
    )
    drhs = grad_dot_general(
        dg_res.lhs,
        dg_res.rhs.quant_grad,
        cfg.drhs,
        True,
    )
    # fwd_dimension_numbers are marked as nondiff_argnums instead of returning
    # None as grad to it. This is because it is a tuple of Python integers
    # that cannot be traced by Jax.
    return (dlhs, drhs, None, None, None)

  vjp = jax.custom_vjp(make_fwd(False), nondiff_argnums=(4,))
  vjp.defvjp(make_fwd(True), vjp_bwd)

  return vjp


def make_dot_general(cfg: Optional[config.DotGeneral]):
  """Makes quantized lax.dot_general replacement with attached gradients."""
  if cfg is None:
    return jax.lax.dot_general

  dg = _dot_general_raw_attach_gradient()

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

    out, _ = dg(
        lhs=lhs,
        rhs=rhs,
        lhs_qt=None,
        rhs_qt=None,
        dimension_numbers=dimension_numbers,
        cfg=cfg,
    )
    return out

  return ret_dg
