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
# pylint: disable=g-importing-member
import copy
import functools
from typing import Iterable
from typing import Optional, Union
from aqt.jax.v2 import aqt_dot_general
from aqt.jax.v2 import aqt_tensor
from aqt.jax.v2 import config
from aqt.jax.v2 import tiled_dot_general
from aqt.jax.v2.flax.utils import QuantMode
import flax.linen as nn
import jax
import jax.numpy as jnp


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
      qvalue = self.qvalue.value
      # TODO(b/325626080): Remove the optional logic.
      if self.q_dtype == jnp.int4:
        qvalue = qvalue.astype(jnp.int4)
      return aqt_tensor.QTensor(
          qvalue,
          scale=None,
          scale_t=[self.scale_t.value],
          dequant_dtype=None,  # Rely on dg output dtype for dequant
      )
    else:
      assert False, 'Unknown quant mode.'

  def set(self, inputs: aqt_tensor.QTensor) -> None:
    # TODO(b/325626080): Uncomment the assert.
    # assert inputs.qvalue.dtype == self.q_dtype, (
    #     f'Freezer got a QTensor of type {inputs.qvalue.dtype} but expected'
    #     f' {self.q_dtype}.'
    # )
    if self.quant_mode == QuantMode.TRAIN:
      pass
    elif self.quant_mode == QuantMode.CONVERT:
      qvalue = inputs.qvalue
      # TODO(b/325626080): Remove the optional logic.
      if self.q_dtype == jnp.int4:
        assert qvalue.dtype == jnp.int4
        qvalue = qvalue.astype(jnp.int8)
      self.qvalue.value = qvalue
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
  cfg: Optional[aqt_dot_general.DotGeneral] = None
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
  tiling_cfg: Optional[tiled_dot_general.Cfg] = None

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
        q_dtype=cfg.fwd.lhs.quantizer.numerics.get_dtype(),
        q_init=self.lhs_init,
        s_shape=lhs_scale_shape,
        s_init=self.lhs_scale_init,
        quant_collection=self.quant_collection,
    )

    rhs_freezer = Freezer(
        name=self.rhs_var_name,
        quant_mode=rhs_qm,
        q_shape=rhs_shape,
        q_dtype=cfg.fwd.rhs.quantizer.numerics.get_dtype(),
        q_init=self.rhs_init,
        s_shape=rhs_scale_shape,
        s_init=self.rhs_scale_init,
        quant_collection=self.quant_collection,
    )

    prng_name = self.prng_name
    key = self.make_rng(prng_name) if prng_name is not None else None
    cfg = config.set_context(cfg, key, train_step=None)

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

      config.assert_config_validity(cfg)

      # Getter
      lhs_apply_quant_mode = self.lhs_apply_quant_mode
      rhs_apply_quant_mode = self.rhs_apply_quant_mode
      lhs_qt = lhs_freezer.get() if lhs_apply_quant_mode else self.lhs_qtensor
      rhs_qt = rhs_freezer.get() if rhs_apply_quant_mode else self.rhs_qtensor
      out, (out_lhs_qt, out_rhs_qt) = cfg.dg_core(
          lhs=lhs,
          rhs=rhs,
          lhs_qt=lhs_qt,
          rhs_qt=rhs_qt,
          dimension_numbers=dimension_numbers,
      )
      # Setter
      if self.lhs_apply_quant_mode:
        lhs_freezer.set(out_lhs_qt)
      if self.rhs_apply_quant_mode:
        rhs_freezer.set(out_rhs_qt)

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
    if self.tiling_cfg is not None:
      # We integrate tiling here and not on Jax level, so that the Freezers
      # observe tiled shapes.
      aqt_dg = functools.partial(
          tiled_dot_general.tiled_dot_general,
          self.tiling_cfg,
          dot_general=aqt_dg,
      )
    return aqt_dg(
        lhs,
        rhs,
        dimension_numbers,
        precision,
        preferred_element_type,
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
