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
import dataclasses
from typing import Optional, Callable

import flax.struct
import jax
from jax import lax
import jax.numpy as jnp
import numpy as onp


@dataclasses.dataclass
class TensorConfig:
  """Configuration of quantization of one tensor or one side of tensor op."""

  bits: Optional[int]
  calib_shared_axes: Optional[list[int]]
  preserve_zero: bool
  bound: Optional[float]
  bound_stop_grad: bool
  # false = map max val on the end of the last bucket
  # true = map max val on the middle of the last
  preserve_max_val: bool
  clip: bool
  round: bool
  noise_fn: Optional[
      Callable[[tuple[int, ...], jax.random.KeyArray], jnp.ndarray]
  ]
  # Round up the calibration to power of 2 (po2).
  po2_scale: bool

  @classmethod
  def make(cls, bits: Optional[int]) -> 'TensorConfig':
    pz = False if bits == 1 else True

    return TensorConfig(
        bits=bits,
        calib_shared_axes=None,
        preserve_zero=pz,
        bound=None,
        bound_stop_grad=True,
        preserve_max_val=False,
        clip=True,
        round=True,
        noise_fn=None,
        po2_scale=False,
    )


@dataclasses.dataclass
class DotGeneralRawConfig:
  """Configuration of quantization of one dot_general without gradient."""
  lhs: TensorConfig
  rhs: TensorConfig
  use_hardware_int8: bool
  # use_fwd_quant is observed when this dot_general is used in gradient.
  # use_fwd_quant is ignored in forward pass.
  # Whether the gradient should be taken at unquantized wgt/act or quantized.
  use_fwd_quant: bool

  @classmethod
  def make(cls, lhs_bits=None, rhs_bits=None) -> 'DotGeneralRawConfig':
    """Create quantization configs for input matrices to a matmul."""
    return DotGeneralRawConfig(
        lhs=TensorConfig.make(lhs_bits),
        rhs=TensorConfig.make(rhs_bits),
        use_hardware_int8=False,
        use_fwd_quant=True,
    )


@dataclasses.dataclass
class DotGeneralConfig:
  fwd: DotGeneralRawConfig
  dlhs: DotGeneralRawConfig
  drhs: DotGeneralRawConfig

  @classmethod
  def make(cls, lhs_bits=None, rhs_bits=None) -> 'DotGeneralConfig':
    """Create quantization configs for input matrices to a matmul."""
    return DotGeneralConfig(
        fwd=DotGeneralRawConfig.make(lhs_bits, rhs_bits),
        dlhs=DotGeneralRawConfig.make(),
        drhs=DotGeneralRawConfig.make(),
    )


def make_config_conv_general_dilated(
    spatial_dimensions=2,
    lhs_bits=None,
    rhs_bits=None,
) -> DotGeneralRawConfig:
  config = DotGeneralRawConfig.make(lhs_bits, rhs_bits)
  # Hardcoding flax assumptions.
  if config.lhs:
    config.lhs.calib_shared_axes = list(range(1, spatial_dimensions + 2))
  if config.rhs:
    config.rhs.calib_shared_axes = list(range(0, spatial_dimensions + 2 - 1))
  return config


# The arithmetic of scaling, clipping and rounding can be confusing.
# Here I add some documentation to hopefully clarify it.
# Bucket is an interval of rounding after the scaling is applied.
# Facts:
#  - Bucket size is always 1.0.
#  - Middle of the bucket = righ of the bucket - 0.5
#  - If not preserve_zero then
#    - bucket ends align with integers.
#    - bucket center align with integers+0.5 .
#  - If preserve_zero then
#    - bucket ends align with intigers+0.5.
#    - bucket center align with integers.
#  - We have two types of rounding, both mostly unbiased:
#    - round_int0(x) = floor(x+0.5) # rounding to integer
#    - round_int5(x) = floor(x)+0.5 # rounding to integer+0.5
#  - Center of the bucket is presereved by rounding in all cases.

# Let's explore 6 options:
# preserve_zero = False - rounding to x.5. i.e 0.5, 1.5, etc are preserved
#   prec=2
#   buckets: [-2, -1] [-1, 0] [0, 1] [1, 2]
#   bucket centers: -1.5 -0.5 0.5 1.5
#     preserve_max_val = False
#     we map largest value to 2.0 (mapping to the end of largest bucket)
#     preserve_max_val = True
#     we map largest value to 1.5
#   prec=1
#   bucket centers: -0.5 0.5
#   buckets: [-1, 0] [0, 1]
#     preserve_max_val = False
#     we map largest value to 1.0
#     preserve_max_val = True
#     we map largest value to 0.5
# preserve_zero = True - rounding to x.0 i.e 0.0, 1.0, 2.0, etc are preserved
#   prec=2
#   buckets: [-1.5, -0.5] [-0.5, 0.5] [0.5, 1.5]
#   bucket centers: -1, 0, 1
#     preserve_max_val = False
#     we map largest value to 1.5 (mapping to the end of largest bucket)
#     preserve_max_val = True
#     we map largest value to 1.0

# Summary in the table.
# preserve_zero, preserve_max_val, max_val_mapped_to, clipping_formula
## True , False , 2^(n-1) - 0.5, round_int0(clip(x, 2^(n-1) - 0.5 - eps))
# True , True  , 2^(n-1) - 1.0, round_int0(clip(x, 2^(n-1) - 0.5 - eps))
# False, False , 2^(n-1)      , round_int5(clip(x, 2^(n-1) - 0.0 - eps))
# False, True  , 2^(n-1) - 0.5, round_int5(clip(x, 2^(n-1) - 0.0 - eps))
#
# Clipping is there only to round all the buckets beyond prec to the biggest
# bucket. (we will have a separate method for gradients)
#
# We need eps>0.0 so that the fwd pass of round_int0(x) = floor(x+0.5) does not
# have an edge condition on -(2^(n-1) - 0.5). That would add additional bucket.
# eps can be anywhere in (0, 1.0) for correctness (sharp inequalities).
# We choose eps=0.5
# However this messes with the gradient.
# Also reducing eps is not good enough for po2 case.


@flax.struct.dataclass
class Context:
  key: Optional[jax.random.KeyArray]
  # train_step: int = 0


def _context_split(context: Context) -> tuple[Context, Context]:
  context1, context2 = Context(key=None), Context(key=None)
  if context.key is not None:
    key1, key2 = jax.random.split(context.key)
    context1.key = key1
    context2.key = key2
  return (context1, context2)


def _get_max_value_representation(config: TensorConfig):
  """Largest quantization tensor value is mapped onto 'int' value returned by this function."""
  assert config.bits is not None
  assert config.bits <= 22, 'Too many bits, float32 has less precision.'
  clip_bound = 2.0 ** (config.bits - 1)
  if config.preserve_zero:
    clip_bound -= 0.5
  if config.preserve_max_val:
    clip_bound -= 0.5
  return clip_bound


def _get_clip_bound(config: TensorConfig):
  """Returns the clip bound when using integer values."""
  assert config.bits is not None
  assert config.bits <= 22, 'Too many bits, float32 has less precision.'
  clip_bound = 2.0 ** (config.bits - 1)
  if config.preserve_zero:
    clip_bound -= 0.5
  return clip_bound


def _fresh_scale(x, config: TensorConfig) -> jnp.ndarray:
  """Calibration scale."""
  if config is None:
    return jnp.ones((1,) * len(x.shape), dtype=x.dtype)

  # We have 2 sources for contraction axes:
  assert config.calib_shared_axes

  # x_bound is the input range that gets mapped to the integer clip_bound
  # For dynamic quant x_bound = max(x); for static quant x_bound = config.bound
  if config.bound is None:
    x_bound = jnp.max(jnp.abs(x), axis=config.calib_shared_axes, keepdims=True)
  else:
    assert config.bound > 0, 'Static quantization bound should be positive.'
    x_bound = jnp.asarray(config.bound)
  x_bound = jnp.where(x_bound == 0.0, 1.0, x_bound)
  if config.bound_stop_grad:
    x_bound = lax.stop_gradient(x_bound)

  # This is the value that the x_bound is mapped to.
  x_bound_repr = _get_max_value_representation(config)
  new_scale = x_bound_repr / x_bound
  if config.po2_scale:
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


def _make_clip_and_round(config: TensorConfig):
  """Function make_clip_and_round."""
  clip_bound = _get_clip_bound(config)

  def fwd(x, context):
    if config is None:
      return x
    if config.clip:
      # We use eps = 0.5 to make sure that after clipping we, `x` is wholly in
      # the buckets. This does not affect us, because there is _round following.
      if config.round:
        eps = 0.5
      else:
        # If we are not rounding, we don't care about the largest value possible
        # jumping into additional bucket. Because we are not using real ints.
        eps = 0.0
      fwd_clip_bound = clip_bound - eps
      x = jnp.clip(x, -fwd_clip_bound, fwd_clip_bound)
    if config.noise_fn:
      assert context.key is not None, (
          'noise_fn is set, requestic stochastic rounding, but key key was not'
          ' passed.'
      )
      x = x + config.noise_fn(x.shape, context.key)
    if config.round:
      x = _round(x, round_to_halves=not config.preserve_zero)
    return x

  def vjp_fwd(x, context):
    res = (x,)
    return fwd(x, context), res

  def vjp_bwd(fwd_context, res, grad):
    del fwd_context
    (x,) = res
    # This is gradient of clip. For boundary values we will have full graindent.
    ret = (x <= clip_bound) * (x >= -clip_bound) * grad
    return (ret,)

  vjp = jax.custom_vjp(fwd, nondiff_argnums=(1,))
  vjp.defvjp(vjp_fwd, vjp_bwd)
  return vjp


def make_fake_quant(config: TensorConfig):
  def fake_quant(x, context):
    if config.bits is None:
      return x
    scale = _fresh_scale(x, config)
    x = x * scale
    x = _make_clip_and_round(config)(x, context)
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
def _make_dot_general_raw(config: DotGeneralRawConfig, use_fake_quant=False):
  """Makes quantized lax.dot_general replacement."""
  config = copy.deepcopy(config)

  def my_dot_general(
      lhs,
      rhs,
      dimension_numbers,
      context,
      precision=None,
      preferred_element_type=None,
  ):
    orig_lhs = lhs
    orig_rhs = rhs
    # All axes can be partitioned into:
    # - contraction axes (ca)
    # - batch axes (ba)
    # - remaining axes (ra).
    (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers

    context, context_bwd = _context_split(context)
    context_lhs, context_rhs = _context_split(context)
    del context

    if config.lhs.bits is not None:
      config.lhs.calib_shared_axes = config.lhs.calib_shared_axes or lhs_ca
      lhs_scale = _fresh_scale(lhs, config.lhs)
      lhs = lhs * lhs_scale
      lhs = _make_clip_and_round(config.lhs)(lhs, context_lhs)

    if config.rhs.bits is not None:
      config.rhs.calib_shared_axes = config.rhs.calib_shared_axes or rhs_ca
      rhs_scale = _fresh_scale(rhs, config.rhs)
      rhs = rhs * rhs_scale
      rhs = _make_clip_and_round(config.rhs)(rhs, context_rhs)

    if config.use_hardware_int8:
      lhs = lhs.astype(jnp.int8)
      rhs = rhs.astype(jnp.int8)
      preferred_element_type = jnp.int32

    # if lhs.dtype != rhs.dtype:

    out = lax.dot_general(
        lhs,
        rhs,
        dimension_numbers=dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
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

    if config.lhs.bits is not None:
      lhs_scale_t = scale_trans(lhs_scale, lhs_ca, lhs_ba)
      # inserting dummy axes for rhs_ra
      assert len(lhs_scale_t.shape) == len(lhs.shape) - len(lhs_ca)
      start = len(lhs_scale_t.shape)
      end = len(rhs.shape) - len(rhs_ca) - len(rhs_ba) + start
      lhs_dummy_axes = range(start, end)
      lhs_scale_t = 1.0 / jnp.expand_dims(lhs_scale_t, axis=lhs_dummy_axes)
      out = out * lhs_scale_t
    else:
      lhs_scale_t = 1.0
    lhs_res = TensorRes(value=orig_lhs, qvalue=lhs, qvalue_scale=lhs_scale_t)

    if config.rhs.bits is not None:
      rhs_scale_t = scale_trans(rhs_scale, rhs_ca, rhs_ba)
      start = len(rhs_ba)
      end = len(lhs.shape) - len(lhs_ca) - len(lhs_ba) + start
      rhs_dummy_axes = range(start, end)
      rhs_scale_t = jnp.expand_dims(rhs_scale_t, axis=rhs_dummy_axes)
      rhs_scale_t = 1.0 / rhs_scale_t
      out = out * rhs_scale_t
    else:
      rhs_scale_t = 1.0
    rhs_res = TensorRes(value=orig_rhs, qvalue=rhs, qvalue_scale=rhs_scale_t)

    res = DotGeneralRes(context_bwd=context_bwd, lhs=lhs_res, rhs=rhs_res)
    return out, res

  def fq_dot_general(
      lhs,
      rhs,
      dimension_numbers,
      context,
      precision=None,
      preferred_element_type=None,
  ):
    msg = (
        'use_fake_quant mode is used in tests and it is exactly equal when'
        ' po2_scale == True; Did you forget to set it?'
    )
    assert config.lhs.po2_scale, msg
    assert config.rhs.po2_scale, msg

    context, context_bwd = _context_split(context)
    context_lhs, context_rhs = _context_split(context)
    del context
    lhs_fq = make_fake_quant(config.lhs)(lhs, context_lhs)
    rhs_fq = make_fake_quant(config.rhs)(rhs, context_rhs)
    ret = jax.lax.dot_general(
        lhs_fq,
        rhs_fq,
        dimension_numbers,
        precision,
        preferred_element_type=preferred_element_type,
    )
    res = DotGeneralRes(
        context_bwd=context_bwd,
        lhs=TensorRes(value=lhs, qvalue=lhs_fq, qvalue_scale=1.0),
        rhs=TensorRes(value=rhs, qvalue=rhs_fq, qvalue_scale=1.0),
    )
    return ret, res

  if use_fake_quant:
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

  def vjp_bwd(
      fwd_dims,
      fwd_context,
      precision,
      preferred_element_type,
      res: DotGeneralRes,
      g,
  ):
    # fwd_context contains the key that was captured in vjp_fwd.
    # It was already used there and we should not use it here again.
    # If we need a key, we should use one passed into res parameter.
    del fwd_context
    def ranges_like(*xs, start=0):
      for x in xs:
        yield tuple(range(start, start + len(x)))
        start += len(x)

    def grad_dot_general(
        y_res: TensorRes, dot_general, y_is_lhs, context, use_fwd_quant
    ):
      y_ndim = y_res.value.ndim

      (x_ca, y_ca), (x_ba, y_ba) = fwd_dims
      if y_is_lhs:
        (y_ca, x_ca) = (x_ca, y_ca)
        (y_ba, x_ba) = (x_ba, y_ba)
      g_ndim = g.ndim - y_ndim + len(x_ba) + 2 * len(x_ca)
      x_ra = tuple(i for i in range(g_ndim) if i not in set(x_ca + x_ba))
      y_ra = tuple(i for i in range(y_ndim) if i not in set(y_ca + y_ba))
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
      out, _ = dot_general(
          gv, yv, dims, context, precision, preferred_element_type,
      )

      x_ca_sorted_by_y = tuple(onp.take(x_ca, onp.argsort(y_ca)))
      out_axes = tuple(onp.argsort(x_ba + x_ra + x_ca_sorted_by_y))
      return jax.lax.transpose(out, out_axes)

    context1, context2 = _context_split(res.context_bwd)
    dlhs = grad_dot_general(
        res.rhs, dlhs_dot_general_raw, False, context1, dlhs_use_fwd_quant
    )
    drhs = grad_dot_general(
        res.lhs, drhs_dot_general_raw, True, context2, drhs_use_fwd_quant
    )
    return dlhs, drhs

  def fwd_dot_general_raw_no_res(
      lhs,
      rhs,
      dimension_numbers,
      context,
      precision=None,
      preferred_element_type=None,
  ):
    ret, _ = fwd_dot_general_raw(
        lhs,
        rhs,
        dimension_numbers,
        context,
        precision,
        preferred_element_type,
    )
    return ret

  vjp = jax.custom_vjp(fwd_dot_general_raw_no_res, nondiff_argnums=(2, 3, 4, 5))
  vjp.defvjp(fwd_dot_general_raw, vjp_bwd)
  return vjp


def make_dot_general(config: Optional[DotGeneralConfig]):
  """Makes quantized lax.dot_general replacement with attached gradients."""
  if config is None:
    config = DotGeneralConfig.make()

  dg = _dot_general_raw_attach_gradient(
      fwd_dot_general_raw=_make_dot_general_raw(config.fwd),
      dlhs_dot_general_raw=_make_dot_general_raw(config.dlhs),
      drhs_dot_general_raw=_make_dot_general_raw(config.drhs),
      dlhs_use_fwd_quant=(config.dlhs.use_fwd_quant if config.dlhs else True),
      drhs_use_fwd_quant=(config.drhs.use_fwd_quant if config.drhs else True),
  )

  def ret_dg(
      lhs,
      rhs,
      dimension_numbers,
      context=Context(key=None),
      precision=None,
      preferred_element_type=None,
  ):
    return dg(
        lhs=lhs,
        rhs=rhs,
        dimension_numbers=dimension_numbers,
        context=context,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )

  return ret_dg


def make_conv_general_dilated(config: DotGeneralRawConfig):
  """Makes quantized lax.make_conv_general_dilated replacement."""
  # TODO(lew): Either rename DotGeneralConfig or make a conv-specific config.
  config = copy.deepcopy(config)
  if config is None:
    config = DotGeneralRawConfig.make()

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

    if config.lhs.bits is not None:
      # Flax assumptions.
      assert config.lhs.calib_shared_axes == list(range(1, rank))
      lhs_scale = _fresh_scale(lhs, config.lhs)
      lhs = lhs * lhs_scale
      lhs = _make_clip_and_round(config.lhs)(lhs, None)

    if config.rhs.bits is not None:
      assert config.rhs.calib_shared_axes == list(range(0, rank - 1))
      rhs_scale = _fresh_scale(rhs, config.rhs)
      rhs = rhs * rhs_scale
      rhs = _make_clip_and_round(config.rhs)(rhs, None)

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

    if config.lhs.bits is not None:
      out /= lhs_scale

    if config.rhs.bits is not None:
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
