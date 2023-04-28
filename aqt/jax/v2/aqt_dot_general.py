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

# pylint: disable=g-explicit-bool-comparison

import copy
from aqt.jax.v2 import config
import flax.struct
import jax
from jax import lax
import jax.numpy as jnp
import numpy as onp

# TODO(lew): We want to handle two additional cases:
# - use_fwd_quant with say just weights being int8 and passed to bwd
# - use_fwd_quant but we want to use classic quantization.


@flax.struct.dataclass
class Context:
  key: jax.random.KeyArray | None
  # train_step: int = 0


def _context_split(context: Context) -> tuple[Context, Context]:
  if context.key is not None:
    key1, key2 = jax.random.split(context.key)
    return Context(key=key1), Context(key=key2)
  return Context(key=None), Context(key=None)


def _get_max_value_representation(cfg: config.Tensor):
  """Largest quantization tensor value is mapped onto 'int' value returned by this function."""
  assert cfg.bits is not None
  assert cfg.bits <= 22, 'Too many bits, float32 has less precision.'
  clip_bound = 2.0 ** (cfg.bits - 1)
  if cfg.preserve_zero:
    clip_bound -= 0.5
  if cfg.preserve_max_val:
    clip_bound -= 0.5
  return clip_bound


def _get_clip_bound(cfg: config.Tensor):
  """Returns the clip bound when using integer values."""
  assert cfg.bits is not None
  assert cfg.bits <= 22, 'Too many bits, float32 has less precision.'
  clip_bound = 2.0 ** (cfg.bits - 1)
  if cfg.preserve_zero:
    clip_bound -= 0.5
  return clip_bound


def _fresh_scale(x, cfg: config.Tensor) -> jnp.ndarray:
  """Calibration scale."""
  if cfg is None:
    return jnp.ones((1,) * len(x.shape), dtype=x.dtype)

  # We have 2 sources for contraction axes:
  assert cfg.calib_shared_axes

  # x_bound is the input range that gets mapped to the integer clip_bound
  # For dynamic quant x_bound = max(x); for static quant x_bound = cfg.bound
  if cfg.bound is None:
    x_bound = jnp.max(jnp.abs(x), axis=cfg.calib_shared_axes, keepdims=True)
  else:
    assert cfg.bound > 0, 'Static quantization bound should be positive.'
    x_bound = jnp.asarray(cfg.bound)
  x_bound = jnp.where(x_bound == 0.0, 1.0, x_bound)
  if cfg.bound_stop_grad:
    x_bound = lax.stop_gradient(x_bound)

  # This is the value that the x_bound is mapped to.
  x_bound_repr = _get_max_value_representation(cfg)
  new_scale = x_bound_repr / x_bound
  if cfg.po2_scale:
    # With floor the bigges value (we are using jnp.max) is in the range of
    # clipping and therefore have a correct gradinet.
    new_scale = 2 ** jnp.floor(jnp.log2(new_scale))
  return new_scale


def _round(x, round_to_halves=False):
  """(Mostly) unbiased rounding to either an integer or integer+0.5 ."""
  if round_to_halves:
    return jnp.floor(x) + 0.5
  else:
    # TODO(lew): use RTNE round
    return jnp.floor(x + 0.5)


def _make_clip_and_round(cfg: config.Tensor):
  """Function make_clip_and_round."""
  assert cfg is not None
  clip_bound = _get_clip_bound(cfg)

  def fwd(x, context):
    if cfg.clip:
      # We use eps = 0.5 to make sure that after clipping we, `x` is wholly in
      # the buckets. This does not affect us, because there is _round following.
      if cfg.round:
        eps = 0.5
      else:
        # If we are not rounding, we don't care about the largest value possible
        # jumping into additional bucket. Because we are not using real ints.
        eps = 0.0
      fwd_clip_bound = clip_bound - eps
      x = jnp.clip(x, -fwd_clip_bound, fwd_clip_bound)
    if cfg.noise_fn:
      assert context.key is not None, (
          'noise_fn is set, requestic stochastic rounding, but key key was not'
          ' passed.'
      )
      x = x + cfg.noise_fn(x.shape, context.key)
    if cfg.round:
      x = _round(x, round_to_halves=not cfg.preserve_zero)
    return x

  def vjp_fwd(x, context):
    res = (x,)
    return fwd(x, context), res

  def vjp_bwd(res, grad):
    (x,) = res
    # This is gradient of clip. For boundary values we will have full graindent.
    ret = (x <= clip_bound) * (x >= -clip_bound) * grad
    return (ret, None)

  vjp = jax.custom_vjp(fwd)
  vjp.defvjp(vjp_fwd, vjp_bwd)
  return vjp


def make_fake_quant(cfg: config.Tensor):
  def fake_quant(x, context):
    if cfg.bits is None:
      return x
    scale = _fresh_scale(x, cfg)
    x = x * scale
    x = _make_clip_and_round(cfg)(x, context)
    x = x / scale
    return x

  return fake_quant


@flax.struct.dataclass
class TensorRes:
  value: jnp.ndarray
  qvalue: jnp.ndarray
  qvalue_scale: jnp.ndarray | float


@flax.struct.dataclass
class DotGeneralRes:
  context_bwd: Context
  lhs: TensorRes
  rhs: TensorRes


# TODO(lew): Gradient of this function is costly. Optimize.
def _make_dot_general_raw(cfg: config.DotGeneralRaw):
  """Makes quantized lax.dot_general replacement."""
  cfg = copy.deepcopy(cfg)

  def my_dot_general(
      lhs,
      rhs,
      dimension_numbers,
      context,
  ):
    # All axes can be partitioned into:
    # - contraction axes (ca)
    # - batch axes (ba)
    # - remaining axes (ra).
    (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers

    context, context_bwd = _context_split(context)
    context_lhs, context_rhs = _context_split(context)
    del context

    qlhs = lhs
    if cfg.lhs.bits is not None:
      cfg.lhs.calib_shared_axes = cfg.lhs.calib_shared_axes or lhs_ca
      lhs_scale = _fresh_scale(qlhs, cfg.lhs)
      qlhs = qlhs * lhs_scale
      qlhs = _make_clip_and_round(cfg.lhs)(qlhs, context_lhs)

    qrhs = rhs
    if cfg.rhs.bits is not None:
      cfg.rhs.calib_shared_axes = cfg.rhs.calib_shared_axes or rhs_ca
      rhs_scale = _fresh_scale(qrhs, cfg.rhs)
      qrhs = qrhs * rhs_scale
      qrhs = _make_clip_and_round(cfg.rhs)(qrhs, context_rhs)

    out = lax.dot_general(
        qlhs.astype(cfg.in_dtype),
        qrhs.astype(cfg.in_dtype),
        dimension_numbers=dimension_numbers,
        preferred_element_type=cfg.preferred_element_type,
    )
    # The axis order in out is as follows: batch, lhs_ra, rhs_ra
    # - batch axes order is uniquely determined by either lhs_ba or rhs_ba
    # - contraction axes ca disappear from the output
    # - order of the remaining axes (ra) is preserved.

    def scale_trans(x, ca, ba):
      for i in ca:
        assert x.shape[i] == 1
      ra = tuple(i for i in range(len(x.shape)) if i not in ba + ca)
      x = jnp.transpose(x, ba + ra + ca)
      # TODO(lew): x = jnp.squeeze(x, axis=range(len(ba+ra): len(x.shape))
      shape_ba = x.shape[: len(ba)]
      shape_ra = x.shape[len(ba) : -len(ca)]
      # Will need to add additional axes (size 1) for the other shape_ra
      x = x.reshape(shape_ba + shape_ra)
      return x

    if cfg.lhs.bits is not None:
      qlhs_scale_t = scale_trans(lhs_scale, lhs_ca, lhs_ba)
      # inserting dummy axes for rhs_ra
      assert len(qlhs_scale_t.shape) == len(lhs.shape) - len(lhs_ca)
      start = len(qlhs_scale_t.shape)
      end = len(rhs.shape) - len(rhs_ca) - len(rhs_ba) + start
      lhs_dummy_axes = range(start, end)
      qlhs_scale_t = 1.0 / jnp.expand_dims(qlhs_scale_t, axis=lhs_dummy_axes)
      out = out * qlhs_scale_t
    else:
      qlhs_scale_t = 1.0
    lhs_res = TensorRes(value=lhs, qvalue=qlhs, qvalue_scale=qlhs_scale_t)

    if cfg.rhs.bits is not None:
      qrhs_scale_t = scale_trans(rhs_scale, rhs_ca, rhs_ba)
      start = len(rhs_ba)
      end = len(lhs.shape) - len(lhs_ca) - len(lhs_ba) + start
      rhs_dummy_axes = range(start, end)
      qrhs_scale_t = jnp.expand_dims(qrhs_scale_t, axis=rhs_dummy_axes)
      qrhs_scale_t = 1.0 / qrhs_scale_t
      out = out * qrhs_scale_t
    else:
      qrhs_scale_t = 1.0
    rhs_res = TensorRes(value=rhs, qvalue=qrhs, qvalue_scale=qrhs_scale_t)

    res = DotGeneralRes(
        context_bwd=context_bwd,
        lhs=lhs_res,
        rhs=rhs_res,
    )
    return out, res

  def fq_dot_general(
      lhs,
      rhs,
      dimension_numbers,
      context,
  ):
    msg = (
        'use_fake_quant mode is used in tests and it is exactly equal when'
        ' po2_scale == True; Did you forget to set it?'
    )
    assert cfg.lhs.po2_scale, msg
    assert cfg.rhs.po2_scale, msg

    context, context_bwd = _context_split(context)
    context_lhs, context_rhs = _context_split(context)
    del context
    lhs_fq = make_fake_quant(cfg.lhs)(lhs, context_lhs)
    rhs_fq = make_fake_quant(cfg.rhs)(rhs, context_rhs)
    ret = jax.lax.dot_general(
        lhs_fq,
        rhs_fq,
        dimension_numbers,
    )
    res = DotGeneralRes(
        context_bwd=context_bwd,
        lhs=TensorRes(value=lhs, qvalue=lhs_fq, qvalue_scale=1.0),
        rhs=TensorRes(value=rhs, qvalue=rhs_fq, qvalue_scale=1.0),
    )
    return ret, res

  if cfg.use_fake_quant:
    return fq_dot_general
  else:
    return my_dot_general


def _dot_general_raw_attach_gradient(
    fwd_dot_general_raw,
    dlhs_dot_general_raw,
    drhs_dot_general_raw,
    dlhs_use_fwd_quant=False,
    drhs_use_fwd_quant=False,
):
  """Makes quantized lax.dot_general replacement with attached gradients."""

  def vjp_fwd(
      lhs,
      rhs,
      dimension_numbers,
      context,
  ):
    # Ignore the DotGeneralRes.
    ret, _ = fwd_dot_general_raw(
        lhs,
        rhs,
        dimension_numbers,
        context,
    )
    return ret

  def vjp_bwd(
      fwd_dimension_numbers,
      res: DotGeneralRes,
      g,
  ):
    # fwd_context contains the key that was captured in vjp_fwd.
    # It was already used there and we should not use it here again.
    # If we need a key, we should use one passed into res parameter.
    def ranges_like(*xs, start=0):
      for x in xs:
        yield tuple(range(start, start + len(x)))
        start += len(x)

    def grad_dot_general(
        y_res: TensorRes, dot_general, y_is_lhs, context, use_fwd_quant
    ):
      y_ndim = y_res.value.ndim

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

      if use_fwd_quant:
        gv = g * y_res.qvalue_scale
        yv = y_res.qvalue
      else:
        gv = g
        yv = y_res.value
      out, _ = dot_general(gv, yv, dims, context)

      x_ca_sorted_by_y = tuple(onp.take(x_ca, onp.argsort(y_ca)))
      out_axes = tuple(onp.argsort(tuple(x_ba) + x_ra + x_ca_sorted_by_y))
      return jax.lax.transpose(out, out_axes)

    context1, context2 = _context_split(res.context_bwd)
    dlhs = grad_dot_general(
        res.rhs, dlhs_dot_general_raw, False, context1, dlhs_use_fwd_quant
    )
    drhs = grad_dot_general(
        res.lhs, drhs_dot_general_raw, True, context2, drhs_use_fwd_quant
    )
    return (dlhs, drhs, None)

  vjp = jax.custom_vjp(vjp_fwd, nondiff_argnums=(2,))
  vjp.defvjp(fwd_dot_general_raw, vjp_bwd)
  return vjp


def make_dot_general(cfg: config.DotGeneral | None):
  """Makes quantized lax.dot_general replacement with attached gradients."""
  if cfg is None:
    def ret_lax_dg(
        lhs,
        rhs,
        dimension_numbers,
        precision=None,
        preferred_element_type=None,
        *,
        context=Context(key=None),
    ):
      del context
      return jax.lax.dot_general(
          lhs, rhs, dimension_numbers, precision, preferred_element_type
      )

    return ret_lax_dg

  dg = _dot_general_raw_attach_gradient(
      fwd_dot_general_raw=_make_dot_general_raw(cfg.fwd),
      dlhs_dot_general_raw=_make_dot_general_raw(cfg.dlhs),
      drhs_dot_general_raw=_make_dot_general_raw(cfg.drhs),
      dlhs_use_fwd_quant=(cfg.dlhs.use_fwd_quant if cfg.dlhs else True),
      drhs_use_fwd_quant=(cfg.drhs.use_fwd_quant if cfg.drhs else True),
  )

  def ret_dg(
      lhs,
      rhs,
      dimension_numbers,
      precision=None,
      preferred_element_type=None,
      *,
      context=Context(key=None),
  ):
    assert (
        precision is None
    ), f'Precision {precision} requested together with quantization.'
    assert preferred_element_type is None, (
        f'Preferred_element_typerecision {preferred_element_type} requested'
        ' together with quantization.'
    )
    assert lhs.dtype == rhs.dtype, (
        'The only reason we need that, is because we need to determine return'
        ' type.'
    )
    out = dg(
        lhs=lhs,
        rhs=rhs,
        dimension_numbers=dimension_numbers,
        context=context,
    )
    return out.astype(lhs.dtype)

  return ret_dg


def make_conv_general_dilated(cfg: config.DotGeneralRaw):
  """Makes quantized lax.make_conv_general_dilated replacement."""
  # TODO(lew): Either rename DotGeneralConfig or make a conv-specific cfg.
  cfg = copy.deepcopy(cfg)
  if cfg is None:
    cfg = config.DotGeneralRaw.make()

  def my_conv_general_dilated(
      lhs,
      rhs,
      window_strides,
      padding,
      lhs_dilation=None,
      rhs_dilation=None,
      dimension_numbers=None,
      feature_group_count=1,
      batch_group_count=1,
      precision=None,
      preferred_element_type=None,
  ) -> jax.Array:
    msg1 = """
To simplify the code, we currently assume a Flax-particular layout of the data.
This makes sense, because this is the main use-case of this function.
However if there is any other use, we will drop that assumption."""
    rank = len(lhs.shape)
    assert len(rhs.shape) == rank
    assert dimension_numbers is not None, msg1
    assert dimension_numbers.lhs_spec[0:2] == (0, rank - 1), msg1
    assert dimension_numbers.rhs_spec[0:2] == (rank - 1, rank - 2), msg1
    assert dimension_numbers.out_spec[0:2] == (0, rank - 1), msg1
    # In Flax, lhs is the inputs, rhs is the kernel.
    # lhs layout is B, spatials..., Ci
    # rhs layout is: spatials..., Ci, Co
    # out layous it: B, spatials..., Co
    #
    # we need to share these axes: lhs[1:] , rhs[:-1]
    # we have a scale/invscale per: lhs[0] / out[0] and rhs[-1] / out[-1]

    if cfg.lhs.bits is not None:
      # Flax assumptions.
      assert cfg.lhs.calib_shared_axes == list(range(1, rank))
      lhs_scale = _fresh_scale(lhs, cfg.lhs)
      lhs = lhs * lhs_scale
      lhs = _make_clip_and_round(cfg.lhs)(lhs, None)

    if cfg.rhs.bits is not None:
      assert cfg.rhs.calib_shared_axes == list(range(0, rank - 1))
      rhs_scale = _fresh_scale(rhs, cfg.rhs)
      rhs = rhs * rhs_scale
      rhs = _make_clip_and_round(cfg.rhs)(rhs, None)

    out = lax.conv_general_dilated(
        lhs=lhs,
        rhs=rhs,
        window_strides=window_strides,
        padding=padding,
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )

    if cfg.lhs.bits is not None:
      out /= lhs_scale

    if cfg.rhs.bits is not None:
      out /= rhs_scale
    # # Future scale granularity optimization.
    # In 1x1 conv, each pixel (spatial location) can have different scales
    # in 1xN (rows x colums) conv each row can have different scale, but
    # columns need to share the scales ,  because we are adding pixels across.
    #
    # For patch convs we could have separate scales per patch.
    # We don't do that optimization, because there is a  Flax op: ConvLocal
    # using lax.conv_general_dilated_local which uses lax.dot_general.
    #
    # Dilations: If a dilation of LHS is bigger than the total spatial size of
    # RHS, we could use separe (per LHS pixel) scales.
    # The same applies to dilated RHS.
    # We don't do that optimization yet.
    #
    # We can have different scales across different groups.
    # This applies to both feature and batch.
    return out

  return my_conv_general_dilated
