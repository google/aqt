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
"""Flax layer for AQT injection."""

# pylint: disable=unnecessary-lambda

import copy
import enum
import functools
from typing import Iterable
from typing import Optional, Union
from aqt.jax.v2 import aqt_dot_general
from aqt.jax.v2 import aqt_tensor
from aqt.jax.v2 import calibration
from aqt.jax.v2 import config
from aqt.jax.v2.numerics import int_numerics
from aqt.jax.v2.numerics import no_numerics
import flax.linen as nn
import jax
import jax.numpy as jnp


class QuantMode(enum.Enum):
  TRAIN = 1
  CONVERT = 2
  SERVE = 3


class Freezer(nn.Module):
  """Identity function that can freeze its input.

  On default it is an identity function that saves the input in a variable.
  In 'quant_mode=QuantMode.Serve' mode, ignores the input and returns the frozen
  value. It is usefult to implement 'constant folding' and put quantized weights
  and scales in the checkpoint for serving. Specifically:

  self.get() returns None when quant_mode=TRAIN or CONVERT, returns variable
  when quant_mode=SERVE.
  self.set() does nothing when quant_mode=TRAIN or SERVE, creates and stores
  quantized tensor when quant_mode=CONVERT.
  """

  quant_collection: str
  quant_mode: QuantMode
  q_shape: Iterable[int]
  q_dtype: jnp.dtype
  q_init: nn.initializers.Initializer
  s_shape: Iterable[int]
  s_init: nn.initializers.Initializer

  def setup(self):
    mode = self.quant_mode
    if mode == QuantMode.SERVE or mode == QuantMode.CONVERT:
      collection = self.quant_collection
      q_init = self.q_init
      q_shape = self.q_shape
      q_dtype = self.q_dtype
      s_init = self.s_init
      s_shape = self.s_shape
      # TODO(lew): Store whole QTensor?
      # We could have created one self.variable whose value is a QTensor,
      # but we are unsure how this would complicate the init function,
      # which could potentially be used by adding metadata such as
      # sharding axises, etc.
      self.qvalue = self.variable(collection, 'value', q_init, q_shape, q_dtype)
      self.scale_t = self.variable(collection, 'scale', s_init, s_shape)

  def get(self) -> Optional[aqt_tensor.QTensor]:
    if self.quant_mode == QuantMode.TRAIN:
      return None
    elif self.quant_mode == QuantMode.CONVERT:
      return None
    elif self.quant_mode == QuantMode.SERVE:
      return aqt_tensor.QTensor(
          self.qvalue.value,
          scale=None,
          scale_t=[self.scale_t.value],
          dequant_dtype=None,  # Rely on dg output dtype for dequant
      )
    else:
      assert False, 'Unknown quant mode.'

  def set(self, inputs: aqt_tensor.QTensor) -> None:
    if self.quant_mode == QuantMode.TRAIN:
      pass
    elif self.quant_mode == QuantMode.CONVERT:
      self.qvalue.value = inputs.qvalue
      assert inputs.scale_t is not None and len(inputs.scale_t) == 1
      self.scale_t.value = inputs.scale_t[0]
    elif self.quant_mode == QuantMode.SERVE:
      # TODO(lew): Optionally compare stored and served value.
      pass
    else:
      assert False, 'Unknown quant mode.'
    return None


class AqtDotGeneral(nn.Module):
  """A layer that can be injected into flax.nn.Dense, etc."""

  cfg: Optional[config.DotGeneral] = None
  prng_name: Optional[str] = 'params'

  # TODO(lew): split out separate class for each side.
  # Quant mode determines whether flax variables are created to store quantized
  # inputs. Refer to the Freezer doc str to see variable creation in each mode.
  lhs_quant_mode: QuantMode = QuantMode.TRAIN
  # apply_quant_mode determines if using Freezer in cfg.get/set_tensor
  lhs_apply_quant_mode: bool = True
  lhs_init: nn.initializers.Initializer = jnp.zeros
  lhs_scale_init: nn.initializers.Initializer = jnp.zeros
  lhs_var_name: str = 'qlhs'
  lhs_qtensor: Optional[aqt_tensor.QTensor] = None

  rhs_quant_mode: QuantMode = QuantMode.TRAIN
  rhs_apply_quant_mode: bool = True
  rhs_init: nn.initializers.Initializer = jnp.zeros
  rhs_scale_init: nn.initializers.Initializer = jnp.zeros
  rhs_var_name: str = 'qrhs'
  rhs_qtensor: Optional[aqt_tensor.QTensor] = None

  # If you want use 'params' make sure that there is another mechanism to hide
  # these variables from the optimizer.
  quant_collection: str = 'aqt'

  def make_aqt_dg(
      self,
      lhs_shape,
      rhs_shape,
      dimension_numbers: tuple[Iterable[int], Iterable[int]],
  ):
    if self.cfg is None:
      return jax.lax.dot_general

    cfg = copy.deepcopy(self.cfg)
    lhs_scale_shape = list(lhs_shape)
    rhs_scale_shape = list(rhs_shape)
    (contr, _) = dimension_numbers
    for li, ri in zip(*contr):
      lhs_scale_shape[li] = 1
      rhs_scale_shape[ri] = 1
    lhs_scale = aqt_dot_general._lhs_scale_transpose_to_output(  # pylint: disable=protected-access
        jnp.zeros(lhs_scale_shape), dimension_numbers, lhs_shape, rhs_shape
    )
    assert lhs_scale is not None
    lhs_scale_shape = lhs_scale.shape
    rhs_scale = aqt_dot_general._rhs_scale_transpose_to_output(  # pylint: disable=protected-access
        jnp.zeros(rhs_scale_shape), dimension_numbers, lhs_shape, rhs_shape
    )
    assert rhs_scale is not None
    rhs_scale_shape = rhs_scale.shape
    rhs_qm = self.rhs_quant_mode
    lhs_qm = self.lhs_quant_mode

    lhs_freezer = Freezer(
        name=self.lhs_var_name,
        quant_mode=lhs_qm,
        q_shape=lhs_shape,
        q_dtype=cfg.fwd.lhs.numerics.get_dtype(),
        q_init=self.lhs_init,
        s_shape=lhs_scale_shape,
        s_init=self.lhs_scale_init,
        quant_collection=self.quant_collection,
    )

    rhs_freezer = Freezer(
        name=self.rhs_var_name,
        quant_mode=rhs_qm,
        q_shape=rhs_shape,
        q_dtype=cfg.fwd.rhs.numerics.get_dtype(),
        q_init=self.rhs_init,
        s_shape=rhs_scale_shape,
        s_init=self.rhs_scale_init,
        quant_collection=self.quant_collection,
    )

    prng_name = self.prng_name
    key = self.make_rng(prng_name) if prng_name is not None else None
    cfg = config.set_context(cfg, key, train_step=None)

    dg = aqt_dot_general._dot_general_raw_attach_gradient()  # pylint: disable=protected-access

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

      # TODO(yichizh): asserting xhs dtype only when apply_quant_mode=False
      # and cfg.get_qtensor() is None
      msg = 'AQT is not yet optimized to accept quantized types directly. '
      msg += f'lhs.dtype: {lhs.dtype}, rhs.dtype: {rhs.dtype}'
      assert lhs.dtype in [jnp.bfloat16, jnp.float32, jnp.float16], msg
      assert rhs.dtype in [jnp.bfloat16, jnp.float32, jnp.float16], msg

      # Getter
      lhs_apply_quant_mode = self.lhs_apply_quant_mode
      rhs_apply_quant_mode = self.rhs_apply_quant_mode
      lhs_qt = lhs_freezer.get() if lhs_apply_quant_mode else self.lhs_qtensor
      rhs_qt = rhs_freezer.get() if rhs_apply_quant_mode else self.rhs_qtensor

      out, (out_lhs_qt, out_rhs_qt) = dg(
          lhs=lhs,
          rhs=rhs,
          lhs_qt=lhs_qt,
          rhs_qt=rhs_qt,
          dimension_numbers=dimension_numbers,
          cfg=cfg,
      )

      # TODO(lew): Ideally all QTensors would be always quantized.
      #   Move cast as early as possible.
      def cast(qt: aqt_tensor.QTensor, dtype) -> aqt_tensor.QTensor:
        return qt.replace(qvalue=qt.qvalue.astype(dtype))

      # Setter
      if self.lhs_apply_quant_mode:
        lhs_freezer.set(cast(out_lhs_qt, cfg.fwd.lhs.numerics.get_dtype()))
      if self.rhs_apply_quant_mode:
        rhs_freezer.set(cast(out_rhs_qt, cfg.fwd.rhs.numerics.get_dtype()))

      return out

    return ret_dg

  @nn.compact
  def __call__(
      self,
      lhs,
      rhs,
      dimension_numbers,
      precision,
      preferred_element_type=None,
  ):
    aqt_dg = self.make_aqt_dg(lhs.shape, rhs.shape, dimension_numbers)
    return aqt_dg(
        lhs,
        rhs,
        dimension_numbers,
        precision,
        preferred_element_type=preferred_element_type,
    )


class AqtEinsum(nn.Module):
  """Quantized Einsum class for model injection."""

  cfg: Optional[config.DotGeneral] = None
  prng_name: Optional[str] = 'params'

  # TODO(lew): split out separate class for each side.
  lhs_quant_mode: QuantMode = QuantMode.TRAIN
  lhs_init: nn.initializers.Initializer = jnp.zeros
  lhs_scale_init: nn.initializers.Initializer = jnp.zeros
  lhs_var_name: str = 'qlhs'

  rhs_quant_mode: QuantMode = QuantMode.TRAIN
  rhs_init: nn.initializers.Initializer = jnp.zeros
  rhs_scale_init: nn.initializers.Initializer = jnp.zeros
  rhs_var_name: str = 'qrhs'

  # If you want use 'params' make sure that there is another mechanism to hide
  # these variables from the optimizer.
  quant_collection: str = 'aqt'

  @nn.compact
  def __call__(
      self,
      eqn,
      lhs_g: Union[jnp.ndarray, aqt_tensor.QTensor],
      rhs_g: Union[jnp.ndarray, aqt_tensor.QTensor],
  ):
    cfg = self.cfg
    lhs_is_qt = isinstance(lhs_g, aqt_tensor.QTensor)
    rhs_is_qt = isinstance(rhs_g, aqt_tensor.QTensor)
    msg = 'Aqt config is None but inputs to AqtEinsum are QTensor.'
    assert not ((lhs_is_qt or rhs_is_qt) and cfg is None), msg
    # when inputs are qtensor, xhs_in is a dummy input that will be consumed by
    # lax einsum API, but it is not used for computation in aqt_dg because it
    # will be overwritten by get_tensor()
    # TODO(lew): We can pass QTensor to lax_numpy._einsum if we add some
    # specific methods to QTensor.
    lhs_in = jnp.zeros_like(lhs_g.qvalue) if lhs_is_qt else lhs_g
    rhs_in = jnp.zeros_like(rhs_g.qvalue) if rhs_is_qt else rhs_g

    # Set the types of dummy input to the same as original input, to prevent it
    # from being rejected by assertions in aqt_dot_general.py, line 522-526 and
    # 414.
    # TODO: b/322111904 - Handle this in more proper way.
    lhs_in, rhs_in = nn.dtypes.promote_dtype(lhs_in, rhs_in)

    # yes_swap = whether einsum swaps [lhs,rhs] when passing them to dot_general
    einsum = functools.partial(aqt_dot_general.einsum, eqn=eqn)
    a = jax.make_jaxpr(einsum)(lhs=lhs_in, rhs=rhs_in)
    [lhs_g_id, rhs_g_id] = a.eqns[0].invars
    [lhs_l_id, rhs_l_id] = a.jaxpr.invars
    not_swap = lhs_g_id == lhs_l_id and rhs_g_id == rhs_l_id
    yes_swap = lhs_g_id == rhs_l_id and rhs_g_id == lhs_l_id
    assert not_swap != yes_swap

    prng_name = self.prng_name

    lhs_quant_mode = self.lhs_quant_mode
    lhs_init = self.lhs_init
    lhs_scale_init = self.lhs_scale_init
    lhs_var_name = self.lhs_var_name
    lhs_qtensor = lhs_g if lhs_is_qt else None

    rhs_quant_mode = self.rhs_quant_mode
    rhs_init = self.rhs_init
    rhs_scale_init = self.rhs_scale_init
    rhs_var_name = self.rhs_var_name
    rhs_qtensor = rhs_g if rhs_is_qt else None

    quant_collection = self.quant_collection

    if yes_swap:
      if cfg is not None:
        cfg = copy.deepcopy(cfg)
        cfg.fwd.lhs, cfg.fwd.rhs = cfg.fwd.rhs, cfg.fwd.lhs
        cfg.dlhs, cfg.drhs = cfg.drhs, cfg.dlhs
      lhs_quant_mode, rhs_quant_mode = rhs_quant_mode, lhs_quant_mode
      lhs_init, rhs_init = rhs_init, lhs_init
      lhs_scale_init, rhs_scale_init = rhs_scale_init, lhs_scale_init
      lhs_var_name, rhs_var_name = rhs_var_name, lhs_var_name
      lhs_is_qt, rhs_is_qt = rhs_is_qt, lhs_is_qt
      lhs_qtensor, rhs_qtensor = rhs_qtensor, lhs_qtensor

    aqt_dg = AqtDotGeneral(
        cfg=cfg,
        prng_name=prng_name,
        lhs_quant_mode=lhs_quant_mode,
        # when passing pre-computed qtensor as inputs, apply_quant_mode flag
        # should be set to False so that Freezer will not be set to overwrite
        # the qtensor passed to dg.
        lhs_apply_quant_mode=not lhs_is_qt,  # Freezer not used if lhs is qt
        lhs_init=lhs_init,
        lhs_scale_init=lhs_scale_init,
        lhs_var_name=lhs_var_name,
        lhs_qtensor=lhs_qtensor,
        rhs_quant_mode=rhs_quant_mode,
        rhs_apply_quant_mode=not rhs_is_qt,  # Freezer not used if rhs is qt
        rhs_init=rhs_init,
        rhs_scale_init=rhs_scale_init,
        rhs_var_name=rhs_var_name,
        rhs_qtensor=rhs_qtensor,
        quant_collection=quant_collection,
    )
    return einsum(lhs=lhs_in, rhs=rhs_in, dg=aqt_dg)


def config_v4(
    *,
    fwd_bits: Optional[int] = 8,
    dlhs_bits: Optional[int] = 8,
    drhs_bits: Optional[int] = None,
    # The dummy static bound flag is for performance benchmarking.
    use_dummy_static_bound: bool = False,
    rng_type: str = 'jax.uniform',  # 'custom-1'
    dlhs_local_aqt: Optional[config.LocalAqt] = None,
    drhs_local_aqt: Optional[config.LocalAqt] = None,
    fwd_accumulator_dtype: ... = jnp.int32,
    dlhs_accumulator_dtype: ... = jnp.int32,
    drhs_accumulator_dtype: ... = None,
) -> config.DotGeneral:
  """Version 4 of user-visible AQT config."""

  def tensor_config(bits: Optional[int]) -> config.Tensor:
    assert bits is None or bits >= 2, 'Need at least 2 bits.'
    if bits is None:
      numerics = no_numerics.NoNumerics()
    else:
      numerics = int_numerics.IntNumerics(
          bits=bits,
          preserve_zero=True,
          preserve_max_val=False,
          clip=True,
          round=True,
          noise_fn=None,
          clip_gradient=False,  # Can be False when using abs-max scaling.
          dtype=jnp.int8 if 2 <= bits <= 8 else None,
      )

    return config.Tensor(
        numerics=numerics,
        calib_shared_axes=None,
        scale_stop_grad=True,
        calibration=calibration.AbsMaxCalibration(),
        po2_scale=False,
        use_fwd_quant=None,
        context=config.Context(key=None, train_step=None),
        dequant_mode=config.DequantMode.OUTPUT,
    )

  def dg_raw_config(
      lhs_bits, rhs_bits, jax_scope_name, local_aqt=None
  ) -> config.DotGeneralRaw:
    lhs_cfg = tensor_config(lhs_bits)
    rhs_cfg = tensor_config(rhs_bits)
    if (
        True  # Just to format lines below
        and lhs_bits is not None
        and rhs_bits is not None
        and lhs_bits <= 8
        and rhs_bits <= 8
    ):
      dg_accumulator_dtype = jnp.int32
    else:
      # None determines the dtype on the fly in aqt_dot_general
      dg_accumulator_dtype = None

    return config.DotGeneralRaw(
        lhs=lhs_cfg,
        rhs=rhs_cfg,
        dg_accumulator_dtype=dg_accumulator_dtype,
        local_aqt=local_aqt,
        jax_scope_name=jax_scope_name,
    )

  cfg = config.DotGeneral(
      fwd=dg_raw_config(fwd_bits, fwd_bits, jax_scope_name='aqt_fwd'),
      dlhs=dg_raw_config(
          dlhs_bits,
          dlhs_bits,
          jax_scope_name='aqt_dlhs',
          local_aqt=dlhs_local_aqt,
      ),
      drhs=dg_raw_config(
          drhs_bits,
          drhs_bits,
          jax_scope_name='aqt_drhs',
          local_aqt=drhs_local_aqt,
      ),
  )

  cfg.dlhs.rhs.use_fwd_quant = False
  cfg.drhs.rhs.use_fwd_quant = False

  # Typically we have (but I don't know if it is guraranteed):
  # - vjp_lhs_stochastic_rounding is referring to the gradient and
  # - vjp_rhs_stochastic_rounding is referring to the activations/weights.
  config.set_stochastic_rounding(
      cfg,
      vjp_lhs_stochastic_rounding=True,
      vjp_rhs_stochastic_rounding=False,
      implementation=rng_type,
  )

  if use_dummy_static_bound:
    config.set_static_bound(cfg, 1.0)

  config.set_accumulator_dtype(
      cfg,
      fwd_dtype=fwd_accumulator_dtype,
      dlhs_dtype=dlhs_accumulator_dtype,
      drhs_dtype=drhs_accumulator_dtype,
  )
  assert cfg.fwd.local_aqt is None, 'local_aqt is not yet supported in fwd.'

  return cfg
