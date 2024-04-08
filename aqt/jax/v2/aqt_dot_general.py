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

import abc
import enum
from typing import Any, Optional, Sequence, Union

from aqt.jax.v2 import aqt_quantizer
from aqt.jax.v2 import aqt_tensor
from aqt.jax.v2 import transpose
from aqt.jax.v2 import utils
import jax
from jax import lax
from jax._src.numpy import lax_numpy
import jax.numpy as jnp
import numpy as onp
from typing_extensions import Self  # for python version < 3.11


dtypes_allowed_for_int32_accum = [jnp.int4, jnp.int8]


class CalibrationMode(enum.Enum):
  """Calibration axis modes."""

  CONTRACTING_AXIS = 1
  REMAINING_AXIS = 2


class DequantMode(enum.Enum):
  """Dequant modes."""

  # Multiply output of dot_general by the transposed scale
  # Compatible with CONTRACTING_AXIS.
  OUTPUT = 1
  # Multiply QTensor.qvalue by untransposed QTensor.scale before
  # dot_general (a.k.a. FakeQuant )
  # Compatible with both CalibrationMode.
  THIS_INPUT = 2
  # Multiply other argument of dot general by appropriately transposed scale.
  # Compatible with REMAINING_AXIS.
  OTHER_INPUT = 3


@utils.flax_slots_dataclass
class Tensor:
  """Configuration of quantization of one tensor or one side of tensor op."""

  # Controls at what value of input tensor should be used.
  # Setting it to True, but not quantizing fwd pass will assert-fail.
  use_fwd_quant: Optional[bool] = utils.static_field()
  # Dequantization mode.
  dequant_mode: DequantMode = utils.static_field()
  # Calibration axis mode.
  calibration_mode: CalibrationMode = utils.static_field()

  @classmethod
  def make(cls, *args, **kwargs) -> Self:
    return tensor_make(*args, **kwargs)


@utils.flax_slots_dataclass
class LocalAqt:
  contraction_axis_shard_count: int = utils.static_field()


def tensor_make() -> 'Tensor':
  """Makes config.Tensor."""
  return Tensor(
      use_fwd_quant=None,
      dequant_mode=DequantMode.OUTPUT,
      calibration_mode=CalibrationMode.CONTRACTING_AXIS,
  )


def dot_general_raw_make(
    lhs_bits=None,
    rhs_bits=None,
    local_aqt=None,
    jax_scope_name='aqt',
) -> 'DotGeneralRaw':
  """Create quantization configs for input matrices to a matmul."""
  lhs_cfg = tensor_make()
  rhs_cfg = tensor_make()

  # Binary uses 0.5 right now.
  if (
      lhs_bits is not None
      and rhs_bits is not None
      and 2 <= lhs_bits <= 8
      and 2 <= rhs_bits <= 8
  ):
    dg_accumulator_dtype = jnp.int32
  else:
    dg_accumulator_dtype = None

  lhs = aqt_quantizer.quantizer_make(lhs_bits)
  rhs = aqt_quantizer.quantizer_make(rhs_bits)
  dg_quantizer = DefaultDotGeneralQuantizer(lhs=lhs, rhs=rhs)

  return DotGeneralRaw(
      lhs=lhs_cfg,
      rhs=rhs_cfg,
      dg_quantizer=dg_quantizer,
      dg_accumulator_dtype=dg_accumulator_dtype,
      local_aqt=local_aqt,
      jax_scope_name=jax_scope_name,
  )


def conv_general_dilated_make(
    spatial_dimensions=2,
    lhs_bits: Optional[int] = None,
    rhs_bits: Optional[int] = None,
) -> 'DotGeneralRaw':
  """Create quantization config conv_general_dilated."""
  config = dot_general_raw_make(lhs_bits, rhs_bits)
  # Hardcoding flax assumptions.
  lhs_calib_shared_axes = (
      list(range(1, spatial_dimensions + 2)) if config.lhs else None
  )
  rhs_calib_shared_axes = (
      list(range(0, spatial_dimensions + 2 - 1)) if config.rhs else None
  )

  assert isinstance(config.dg_quantizer, DefaultDotGeneralQuantizer)
  config.dg_quantizer.lhs.calib_shared_axes = lhs_calib_shared_axes
  config.dg_quantizer.rhs.calib_shared_axes = rhs_calib_shared_axes

  return config


def dot_general_make(
    lhs_bits: Optional[int] = None,
    rhs_bits: Optional[int] = None,
    bwd_bits: Optional[int] = None,
    use_fwd_quant: bool = True,
    dlhs_local_aqt=None,
    drhs_local_aqt=None,
) -> 'DotGeneral':
  """Create quantization configs for input matrices to a matmul."""
  fwd = dot_general_raw_make(lhs_bits, rhs_bits, jax_scope_name='aqt_fwd')
  dlhs = dot_general_raw_make(
      bwd_bits, bwd_bits, local_aqt=dlhs_local_aqt, jax_scope_name='aqt_dlhs'
  )
  drhs = dot_general_raw_make(
      bwd_bits, bwd_bits, local_aqt=drhs_local_aqt, jax_scope_name='aqt_drhs'
  )
  cfg = DotGeneral(fwd=fwd, dlhs=dlhs, drhs=drhs)

  # Surprising: lhs quantization determines what drhs can do.
  if lhs_bits is not None:
    # Only rhs is accepting MultiTensor.
    cfg.drhs.rhs.use_fwd_quant = use_fwd_quant
  if rhs_bits is not None:
    cfg.dlhs.rhs.use_fwd_quant = use_fwd_quant
  assert cfg.fwd.local_aqt is None, 'local_aqt is not yet supported in fwd.'
  return cfg


# ------------------------------------------------------------------------------


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


@utils.flax_slots_dataclass
class DotGeneralQuantizer(abc.ABC):
  """Abstract class for dot_general quantizer."""

  @abc.abstractmethod
  def __call__(
      self,
      lhs_quantization_info: tuple[jax.Array, Sequence[int]],
      rhs_quantization_info: tuple[jax.Array, Sequence[int]],
  ) -> tuple[
      tuple[aqt_tensor.QTensor, aqt_tensor.GradientFn],
      tuple[aqt_tensor.QTensor, aqt_tensor.GradientFn],
  ]:
    pass

  @abc.abstractmethod
  def swap_lhs_and_rhs(self) -> None:
    """Swaps lhs and rhs configuration."""
    pass

  @abc.abstractmethod
  def assert_calib_shared_axes_value(
      self,
      lhs_val: Sequence[int] | None,
      rhs_val: Sequence[int] | None,
      msg: str,
  ) -> None:
    """Asserts if calib_shared_axes have certain values."""
    pass

  @abc.abstractmethod
  def set_context(
      self,
      lhs_context: utils.Context,
      rhs_context: utils.Context,
  ) -> None:
    """Sets context for lhs and rhs."""
    pass


@utils.flax_slots_dataclass
class DefaultDotGeneralQuantizer(DotGeneralQuantizer):
  """Default dot_general quantizer."""

  lhs: aqt_quantizer.Quantizer
  rhs: aqt_quantizer.Quantizer

  def __call__(
      self,
      lhs_quantization_info: tuple[jax.Array, Sequence[int]],
      rhs_quantization_info: tuple[jax.Array, Sequence[int]],
  ) -> tuple[
      tuple[aqt_tensor.QTensor, aqt_tensor.GradientFn],
      tuple[aqt_tensor.QTensor, aqt_tensor.GradientFn],
  ]:
    lhs_input, lhs_ca = lhs_quantization_info
    rhs_input, rhs_ca = rhs_quantization_info
    lhs_qt, lhs_quant_grad = self.lhs.quant(lhs_input, calibration_axes=lhs_ca)
    rhs_qt, rhs_quant_grad = self.rhs.quant(rhs_input, calibration_axes=rhs_ca)

    return (lhs_qt, lhs_quant_grad), (rhs_qt, rhs_quant_grad)

  def swap_lhs_and_rhs(self) -> None:
    self.lhs, self.rhs = self.rhs, self.lhs

  def assert_calib_shared_axes_value(
      self,
      lhs_val: Sequence[int] | None,
      rhs_val: Sequence[int] | None,
      msg: str,
  ) -> None:
    assert self.lhs.calib_shared_axes == lhs_val, msg
    assert self.rhs.calib_shared_axes == rhs_val, msg

  def set_context(
      self,
      lhs_context: utils.Context,
      rhs_context: utils.Context,
  ) -> None:
    self.lhs.context = lhs_context
    self.rhs.context = rhs_context


@utils.flax_slots_dataclass
class DotGeneralRaw:
  """Configuration of quantization of one dot_general without gradient."""

  lhs: Tensor
  rhs: Tensor
  dg_quantizer: DotGeneralQuantizer
  dg_accumulator_dtype: Optional[jnp.dtype] = utils.static_field()
  local_aqt: Optional[LocalAqt] = utils.static_field()
  jax_scope_name: str = utils.static_field()

  # Set it to true in order to train with non-None lhs_qt or rhs_qt.
  # Gradient will still flow into lhs_qt and/or rhs_qt, but it may be incorrect.
  # It is a caller responsibility to NOT update these QTensors
  allow_dummy_gradient_into_qtensor: bool = utils.static_field(default=False)
  dot_general: utils.DotGeneralT = utils.static_field(
      default=jax.lax.dot_general
  )

  @classmethod
  def make(cls, *args, **kwargs) -> Self:
    return dot_general_raw_make(*args, **kwargs)

  @classmethod
  def make_conv_general_dilated(cls, *args, **kwargs) -> Self:
    return conv_general_dilated_make(*args, **kwargs)

  def __call__(
      self,
      lhs: jnp.ndarray,
      rhs: Union[jnp.ndarray, MultiTensor],
      # xhs_qt are used in serving.
      lhs_qt: Optional[aqt_tensor.QTensor],
      rhs_qt: Optional[aqt_tensor.QTensor],
      dimension_numbers: jax.lax.DotDimensionNumbers,
  ):
    """A quantized dot_general function without custom gradient."""
    with jax.named_scope(self.jax_scope_name):
      (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
      # TODO(lew):
      #  - Use qx.value with the int type.
      #  - Handle qx.value with the int type in an optimized way.
      #  - Add a "FQ" case we multiply qx.value*qx.value_scale (not transposed).
      #  - Can we carry untransposed scale and transpose here?
      if isinstance(rhs, MultiTensor):
        # We are in gradient code.
        fwd_quantized = rhs.qx.scale_t is not None and len(rhs.qx.scale_t) == 1  # pytype: disable=attribute-error
        expect_fwd_quantized = self.rhs.use_fwd_quant is not None
        msg = (
            'If fwd is quantized, use_fwd_quant should be either True/False;'
            ' otherwise, use_fwd_quant should be None. Misconfiguration: found'
            f' use_fwd_quant is {self.rhs.use_fwd_quant} in bwd, but fwd'
            f' quantization is {fwd_quantized}.'
        )
        assert fwd_quantized == expect_fwd_quantized, msg
        if self.rhs.use_fwd_quant:
          assert fwd_quantized, msg
          # Cast rhs scales to lhs dtype when multiplying with lhs. This is to
          # avoid an unexpected upcast when rhs is float32 but lhs is float16.
          lhs = lhs * rhs.qx.scale_t[0].astype(lhs.dtype)  # pytype: disable=attribute-error
          # rhs qvalue may be integer. It will be quantized again later, so cast
          # its dtype back to dequant dtype.
          # TODO(yichizh): avoid double quantization and evaluate model quality.
          rhs = rhs.qx.qvalue.astype(rhs.qx.dequant_dtype)  # pytype: disable=attribute-error
        else:
          rhs = rhs.x  # pytype: disable=attribute-error
      else:
        assert self.rhs.use_fwd_quant is None, 'cannot set use_fwd_quant in fwd'

      if self.local_aqt is not None:
        local_aqt = self.local_aqt
        factor = local_aqt.contraction_axis_shard_count  # pytype: disable=attribute-error
        msg = 'Custom calib_shared_axes not implemented for local AQT.'
        if isinstance(self.dg_quantizer, DefaultDotGeneralQuantizer):
          self.dg_quantizer.assert_calib_shared_axes_value(None, None, msg)

        def factor_reshape(x, ca, ba):
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

      def _get_calibration_axes(
          tensor_cfg: Tensor,
          ndim: int,
          ca: Sequence[int],
          ba: Sequence[int],
      ) -> Sequence[int]:
        """Computes calibration axes for the given Tensor."""
        match tensor_cfg.calibration_mode:
          case CalibrationMode.REMAINING_AXIS:
            calibration_axes = utils.get_remaining_axes(ndim, ca, ba)
          case CalibrationMode.CONTRACTING_AXIS:
            calibration_axes = ca
          case _:
            raise ValueError(
                f'Unknown calibration mode: {tensor_cfg.calibration_mode}'
            )
        return calibration_axes

      def _postprocess_qtensor(
          input_qtensor: Optional[aqt_tensor.QTensor],
          calculated_qtensor: aqt_tensor.QTensor,
          quant_grad: aqt_tensor.GradientFn,
          tensor_cfg: Tensor,
          transpose_fn: Any,
      ) -> tuple[aqt_tensor.QTensor, str | aqt_tensor.GradientFn]:
        """Compute qtensor from input or input_qtensor."""
        if input_qtensor is not None:
          if not self.allow_dummy_gradient_into_qtensor:
            quant_grad = (
                'Poison. '
                + 'Gradients are not generally expected in serving. '
                + 'Please set allow_dummy_gradient_into_qtensor to True '
                + 'if this is the intended behavior.'
            )
          output_qtensor = input_qtensor
        else:
          output_qtensor = calculated_qtensor
        mode = tensor_cfg.calibration_mode
        if (
            output_qtensor.scale_t is None
            and mode == CalibrationMode.CONTRACTING_AXIS
        ):
          msg = 'scale, scale_t cannot be both unknown'
          assert output_qtensor.scale is not None, msg
          output_qtensor = _get_scale_t(
              qt=output_qtensor,
              transpose_fn=transpose_fn,
              dimension_numbers=dimension_numbers,
              lhs_shape=lhs.shape,
              rhs_shape=rhs.shape,
          )
        return output_qtensor, quant_grad

      lhs_calib_axes = _get_calibration_axes(self.lhs, lhs.ndim, lhs_ca, lhs_ba)
      rhs_calib_axes = _get_calibration_axes(self.rhs, rhs.ndim, rhs_ca, rhs_ba)

      lhs_quantized, rhs_quantized = (
          self.dg_quantizer((lhs, lhs_calib_axes), (rhs, rhs_calib_axes))
      )
      lhs_qt_calculated, lhs_quant_grad = lhs_quantized
      rhs_qt_calculated, rhs_quant_grad = rhs_quantized

      lhs_qt, lhs_quant_grad = _postprocess_qtensor(
          lhs_qt,
          lhs_qt_calculated,
          lhs_quant_grad,
          self.lhs,
          transpose.lhs_scale_transpose_to_output,
      )
      lhs_mt = MultiTensor(x=lhs, qx=lhs_qt)
      lhs_res = TensorRes(mt=lhs_mt, quant_grad=lhs_quant_grad)

      rhs_qt, rhs_quant_grad = _postprocess_qtensor(
          rhs_qt,
          rhs_qt_calculated,
          rhs_quant_grad,
          self.rhs,
          transpose.rhs_scale_transpose_to_output,
      )
      rhs_mt = MultiTensor(x=rhs, qx=rhs_qt)
      rhs_res = TensorRes(mt=rhs_mt, quant_grad=rhs_quant_grad)

      # TODO(lew): mt.x above should be clipped for clipping calibrations

      out = _qtensor_dot_general(
          lhs_qt, rhs_qt, dimension_numbers, self, jnp.promote_types(lhs, rhs)
      )

      out = out.dequant()

      res = DotGeneralRes(
          lhs=lhs_res,
          rhs=rhs_res,
      )
      if self.local_aqt is not None:
        assert len(lhs_ca) == len(rhs_ca)
        if len(lhs_ca) > 0:
          out = jnp.sum(out, axis=0)
        # We are not supporting local AQT in fwd pass, so no res needed.
        res = None
      return out, res


def _qtensor_dot_general(
    lhs_qt: aqt_tensor.QTensor,
    rhs_qt: aqt_tensor.QTensor,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    cfg: ...,  # DotGeneralRaw,
    # dequant_dtype: DType,
    dequant_dtype: jnp.dtype,
) -> aqt_tensor.QTensor:
  """QTensor lax.dot_general replacement."""

  def _maybe_dequant(
      input_qtensor: aqt_tensor.QTensor, tensor_cfg: Tensor
  ) -> jnp.ndarray:
    if tensor_cfg.dequant_mode == DequantMode.THIS_INPUT:
      output = input_qtensor.dequant()
    else:
      output = input_qtensor.qvalue
    return output

  # Dequantize before the lax dg call if in fake quant mode
  lhs_qin = _maybe_dequant(lhs_qt, cfg.lhs)
  rhs_qin = _maybe_dequant(rhs_qt, cfg.rhs)

  dtype_ms = (
      f'Found {cfg.dg_accumulator_dtype=}, {lhs_qin.dtype=} and'
      f' {rhs_qin.dtype=}. Dot general accumulator dtype can only be'
      ' jnp.int32 when both inputs are int8. Otherwise it is recommended to'
      ' be None to let lax.dot_general automatically decide it.'
  )
  if cfg.dg_accumulator_dtype == jnp.int32:
    assert (
        lhs_qin.dtype in dtypes_allowed_for_int32_accum
        and rhs_qin.dtype in dtypes_allowed_for_int32_accum
    ), dtype_ms

  dtypes_can_be_scaled = [jnp.bfloat16, jnp.float32, jnp.float64]
  if cfg.lhs.dequant_mode == DequantMode.OTHER_INPUT:
    assert rhs_qin.dtype in dtypes_can_be_scaled
    for scale in lhs_qt.scale:
      transposed_scale = transpose.lhs_scale_transpose_for_rhs_input(
          scale, dimension_numbers, rhs_qt.shape
      )
      assert isinstance(transposed_scale, jnp.ndarray)  # make pytype quiet
      rhs_qin = rhs_qin * transposed_scale.astype(rhs_qin.dtype)
  if cfg.rhs.dequant_mode == DequantMode.OTHER_INPUT:
    assert lhs_qin.dtype in dtypes_can_be_scaled
    for scale in rhs_qt.scale:
      transposed_scale = transpose.rhs_scale_transpose_for_lhs_input(
          scale, dimension_numbers, lhs_qt.shape
      )
      assert isinstance(transposed_scale, jnp.ndarray)  # make pytype quiet
      lhs_qin = lhs_qin * transposed_scale.astype(lhs_qin.dtype)

  out = cfg.dot_general(
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
      dequant_dtype=dequant_dtype,
  )
  assert out.scale is not None  # pytype help

  if cfg.lhs.dequant_mode == DequantMode.OUTPUT:
    out.scale.extend(lhs_qt.scale_t)
  if cfg.rhs.dequant_mode == DequantMode.OUTPUT:
    out.scale.extend(rhs_qt.scale_t)
  return out


@utils.flax_slots_dataclass
class DotGeneral:
  """Configuration of quantization of dot_general and its gradients."""

  fwd: DotGeneralRaw
  dlhs: DotGeneralRaw
  drhs: DotGeneralRaw

  apply_custom_vjp_on_jax: bool = utils.static_field(default=True)

  @classmethod
  def make(cls, *args, **kwargs) -> Self:
    return dot_general_make(*args, **kwargs)

  def dg_core(
      self,
      lhs: jnp.ndarray,
      rhs: jnp.ndarray,
      lhs_qt: Optional[aqt_tensor.QTensor],
      rhs_qt: Optional[aqt_tensor.QTensor],
      dimension_numbers: lax.DotDimensionNumbers,
  ):
    """dot_general function with expanded API."""
    msg = 'AQT is not yet optimized to accept quantized types directly. '
    msg += f'lhs.dtype: {lhs.dtype}, rhs.dtype: {rhs.dtype}'
    assert lhs.dtype in [jnp.bfloat16, jnp.float32, jnp.float16], msg
    assert rhs.dtype in [jnp.bfloat16, jnp.float32, jnp.float16], msg

    self.assert_config_validity()
    # TODO(dhchoi): Refactor this to make both branches to have the similar
    # functionality of applying custom_vjp.
    if self.apply_custom_vjp_on_jax:
      dg_core_with_custom_grad = jax.custom_vjp(_dg_core, nondiff_argnums=(4,))
      dg_core_with_custom_grad.defvjp(dg_core_vjp_fwd, dg_core_vjp_bwd)
      out, res = dg_core_with_custom_grad(
          lhs, rhs, lhs_qt, rhs_qt, dimension_numbers, self
      )
    else:
      out, res = _dg_core(lhs, rhs, lhs_qt, rhs_qt, dimension_numbers, self)  # pytype: disable=wrong-arg-types
    return out, res

  def __call__(
      self,
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

    out, _ = self.dg_core(
        lhs=lhs,
        rhs=rhs,
        lhs_qt=None,
        rhs_qt=None,
        dimension_numbers=dimension_numbers,
    )
    return out

  def assert_config_validity(self: Self):
    """Asserts if configuration is valid."""

    # use_fwd_quant is not enabled when calibration_axis = remaining_axis.
    msg_fwd_quant = (
        'use_fwd_quant should be set to None when remaining axis are used for'
        ' calibration axis.'
    )

    if self.fwd.rhs.calibration_mode == CalibrationMode.REMAINING_AXIS:
      msg_fwd_quant += (
          f'rhs.calibration_mode: {self.fwd.rhs.calibration_mode},'
          f' dlhs use_fwd_quant: {self.dlhs.rhs.use_fwd_quant}'
      )
      assert self.dlhs.rhs.use_fwd_quant is None, msg_fwd_quant

    if self.fwd.lhs.calibration_mode == CalibrationMode.REMAINING_AXIS:
      msg_fwd_quant += (
          f'lhs.calibration_mode: {self.fwd.lhs.calibration_mode},'
          f' drhs use_fwd_quant: {self.drhs.rhs.use_fwd_quant}'
      )
      assert self.drhs.rhs.use_fwd_quant is None, msg_fwd_quant

    # Check valid combination between calibration_mode and dequant_mode
    unsupported_calibration_dequant_pairs = [
        (DequantMode.OUTPUT, CalibrationMode.REMAINING_AXIS),
        (DequantMode.OTHER_INPUT, CalibrationMode.CONTRACTING_AXIS),
    ]
    msg_mode_mismatch = (
        'Unsupported calibration mode - dequant mode combination '
    )
    for (
        dequant_mode,
        calibration_mode,
    ) in unsupported_calibration_dequant_pairs:
      assert not (
          self.fwd.lhs.calibration_mode == calibration_mode
          and self.fwd.lhs.dequant_mode == dequant_mode
      ), (
          msg_mode_mismatch
          + ' for lhs. calibration_mode:'
          f' {self.fwd.lhs.calibration_mode}, dequant_mode:'
          f' {self.fwd.lhs.dequant_mode}'
      )
      assert not (
          self.fwd.rhs.calibration_mode == calibration_mode
          and self.fwd.rhs.dequant_mode == dequant_mode
      ), (
          msg_mode_mismatch
          + ' for rhs. calibration_mode:'
          f' {self.fwd.rhs.calibration_mode}, dequant_mode:'
          f' {self.fwd.rhs.dequant_mode}'
      )


def _dg_core(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    lhs_qt: Optional[aqt_tensor.QTensor],
    rhs_qt: Optional[aqt_tensor.QTensor],
    dimension_numbers: lax.DotDimensionNumbers,
    cfg: DotGeneral,
):
  out, _ = dg_core_vjp_fwd(lhs, rhs, lhs_qt, rhs_qt, dimension_numbers, cfg)
  return out


# When defining a vjp, all traceable variables must be input arguments of
# both the fwd and bwd function.
# The cfg (DotGeneral) contains the key used for stochastic rounding,
# which are traceable dynamic variables. It needs to be an input argument
# to prevent the jax side effect.
def dg_core_vjp_fwd(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    lhs_qt: Optional[aqt_tensor.QTensor],
    rhs_qt: Optional[aqt_tensor.QTensor],
    dimension_numbers: lax.DotDimensionNumbers,
    cfg: DotGeneral,
):
  """custom_vjp fwd pass."""
  assert (
      lhs.dtype == rhs.dtype
  ), f'Unmatched lhs and rhs dtype: {lhs.dtype} vs {rhs.dtype}'
  ret, res = cfg.fwd(
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
  return ((ret, qret), (res, cfg))


def dg_core_vjp_bwd(
    fwd_dimension_numbers: lax.DotDimensionNumbers,
    res: tuple[Optional[DotGeneralRes], DotGeneral],
    g,
):
  """custom_vjp bwd pass."""
  dg_res, cfg = res
  msg = 'dg_res can only be None in 2nd derivative. It is not yet supported.'
  assert dg_res is not None, msg
  g = g[0]  # g[1] is gradient with respect to qret which we are ignoring.

  def ranges_like(*xs):
    start = 0
    for x in xs:
      yield tuple(range(start, start + len(x)))
      start += len(x)

  def grad_dot_general(
      y_res: TensorRes,
      quant_grad: aqt_tensor.GradientFn,
      dg_raw: DotGeneralRaw,
      y_is_lhs,
  ):
    y_ndim = y_res.mt.x.ndim

    (x_ca, y_ca), (x_ba, y_ba) = fwd_dimension_numbers
    if y_is_lhs:
      (y_ca, x_ca) = (x_ca, y_ca)
      (y_ba, x_ba) = (x_ba, y_ba)
    g_ndim = g.ndim - y_ndim + len(x_ba) + 2 * len(x_ca)
    x_ra = tuple(utils.get_remaining_axes(g_ndim, x_ca, x_ba))
    y_ra = tuple(utils.get_remaining_axes(y_ndim, y_ca, y_ba))
    if y_is_lhs:
      g_ba, g_ca, _ = ranges_like(x_ba, y_ra, x_ra)
    else:
      g_ba, _, g_ca = ranges_like(x_ba, x_ra, y_ra)
    dims = ((g_ca, y_ra), (g_ba, y_ba))

    out, _ = dg_raw(g, y_res.mt, None, None, dims)

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


def make_dot_general(dg: Optional[DotGeneral]):
  # TODO(lew): call warnings.warn("Deprecated")
  if dg is None:
    return jax.lax.dot_general
  else:
    return dg


def assert_config_validity(cfg):
  # TODO(lew): call warnings.warn("Deprecated")
  cfg.assert_config_validity()
