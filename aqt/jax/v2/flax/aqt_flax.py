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

import copy
import enum
from typing import Iterable
from typing import Optional
from aqt.jax.v2 import aqt_dot_general
from aqt.jax.v2 import aqt_tensor
from aqt.jax.v2 import calibration
from aqt.jax.v2 import config
from aqt.jax.v2.numerics import int_numerics
from aqt.jax.v2.numerics import no_numerics
import flax
import flax.linen as nn
import jax
from jax._src.numpy import lax_numpy
import jax.numpy as jnp


class QuantMode(enum.Enum):
  TRAIN = 1
  CONVERT = 2
  SERVE = 3


class Freezer(nn.Module):
  """Identity function that can freeze its input.

  On default it is an identity function that saves the input in a variable.
  In 'use_frozen=True' mode, ignores the input and returns the frozen value. It
  is usefult to implement 'constant folding' and put quantized weights and
  scales in the checkpoint for serving.
  """

  quant_collection: str
  quant_mode: QuantMode
  q_shape: Iterable[int]
  q_dtype: jnp.dtype
  q_init: nn.initializers.Initializer
  s_shape: Iterable[int]
  s_init: nn.initializers.Initializer

  def get(self) -> Optional[aqt_tensor.QTensor]:
    if self.quant_mode == QuantMode.TRAIN:
      return None
    elif self.quant_mode == QuantMode.CONVERT:
      return None
    elif self.quant_mode == QuantMode.SERVE:
      collection = self.quant_collection
      # We could have created one self.variable whose value is a QTensor,
      # but this would complicate the init function, which could potentially
      # be used by adding metadata such as sharding axises, etc.
      qvalue = self.variable(
          collection, 'value', self.q_init, self.q_shape, self.q_dtype
      )
      # TODO(lew): Store whole QTensor?
      scale_t = self.variable(collection, 'scale', self.s_init, self.s_shape)
      # TODO(lew): scale is small, store it instead of using this silly poison.
      poison = jnp.zeros(
          (1, 1, 1, 1, 1, 1, 2, 1, 1, 1)
      )  # hopefully no one uses such shapes
      return aqt_tensor.QTensor(
          qvalue.value, scale=poison, scale_t=scale_t.value
      )
    else:
      assert False, 'Unknown quant mode.'

  def set(self, inputs: aqt_tensor.QTensor) -> None:
    if self.quant_mode == QuantMode.TRAIN:
      pass
    elif self.quant_mode == QuantMode.CONVERT:
      collection = self.quant_collection
      qvalue = self.variable(
          collection, 'value', self.q_init, self.q_shape, self.q_dtype
      )
      scale_t = self.variable(collection, 'scale', self.s_init, self.s_shape)
      qvalue.value = inputs.qvalue
      scale_t.value = inputs.scale_t
    elif self.quant_mode == QuantMode.SERVE:
      # TODO(lew): Optionally compare stored and served value.
      pass
    else:
      assert False, 'Unknown quant mode.'
    return None

  @nn.compact
  def __call__(
      self, inputs: Optional[aqt_tensor.QTensor]
  ) -> Optional[aqt_tensor.QTensor]:
    # TODO(yichizh): Two constraints on Module make the call function necessary:
    # (1) Variables must be created either in setup() or with nn.compact.
    #     We don't want variables in training mode, so nn.compact is better.
    # (2) At most one method in a Module can be wrapped with nn.compact.
    #     so there has to be a fn with the decorator that calls get() and set().
    if inputs is None:
      return self.get()
    else:
      return self.set(inputs)


class AqtDotGeneral(nn.Module):
  """A layer that can be injected into flax.nn.Dense, etc."""

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

  def make_aqt_dg(
      self,
      lhs_shape,
      rhs_shape,
      dimension_numbers: tuple[Iterable[int], Iterable[int]],
  ):
    lhs_scale_shape = list(lhs_shape)
    rhs_scale_shape = list(rhs_shape)
    (contr, _) = dimension_numbers
    for li, ri in zip(*contr):
      lhs_scale_shape[li] = 1
      rhs_scale_shape[ri] = 1
    lhs_scale = aqt_dot_general._lhs_scale_transpose(  # pylint: disable=protected-access
        jnp.zeros(lhs_scale_shape), dimension_numbers, lhs_shape, rhs_shape
    )
    assert lhs_scale is not None
    lhs_scale_shape = lhs_scale.shape
    rhs_scale = aqt_dot_general._rhs_scale_transpose(  # pylint: disable=protected-access
        jnp.zeros(rhs_scale_shape), dimension_numbers, lhs_shape, rhs_shape
    )
    assert rhs_scale is not None
    rhs_scale_shape = rhs_scale.shape

    cfg = copy.deepcopy(self.cfg)
    if cfg is not None:
      rhs_qm = self.rhs_quant_mode
      lhs_qm = self.lhs_quant_mode

      msg = 'The only function that is setting preprocess can be AqtQuantized.'
      assert cfg.fwd.rhs.preprocess is None, msg
      assert cfg.fwd.lhs.preprocess is None, msg
      cfg.fwd.lhs.preprocess = Freezer(
          name=self.lhs_var_name,
          quant_mode=lhs_qm,
          q_shape=lhs_shape,
          q_dtype=cfg.fwd.lhs.numerics.get_dtype(),
          q_init=self.lhs_init,
          s_shape=lhs_scale_shape,
          s_init=self.lhs_scale_init,
          quant_collection=self.quant_collection,
      )
      cfg.fwd.rhs.preprocess = Freezer(
          name=self.rhs_var_name,
          quant_mode=rhs_qm,
          q_shape=rhs_shape,
          q_dtype=cfg.fwd.rhs.numerics.get_dtype(),
          q_init=self.rhs_init,
          s_shape=rhs_scale_shape,
          s_init=self.rhs_scale_init,
          quant_collection=self.quant_collection,
      )
    key = self.make_rng(self.prng_name) if self.prng_name is not None else None
    config.set_context(cfg, key, train_step=None)
    aqt_dg = aqt_dot_general.make_dot_general(cfg)
    return aqt_dg

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


class AqtEinsum(flax.struct.PyTreeNode):
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

  name: Optional[str] = None

  def __call__(self, eqn, lhs_g, rhs_g):
    def einsum(lhs_l, rhs_l, dg=jax.lax.dot_general):
      operands, contractions = lax_numpy._default_poly_einsum_handler(  # pylint: disable=protected-access
          eqn, lhs_l, rhs_l, einsum_call=True, use_blas=True, optimize='optimal'
      )
      contractions = tuple((a, frozenset(b), c) for a, b, c, *_ in contractions)
      return jax.named_call(lax_numpy._einsum, name=eqn)(  # pylint: disable=protected-access
          operands,
          contractions,
          precision=None,
          preferred_element_type=None,
          _dot_general=dg,
      )

    # yes_swap = whether einsum swaps [lhs,rhs] when passing them to dot_general
    a = jax.make_jaxpr(einsum)(lhs_g, rhs_g)
    [lhs_g_id, rhs_g_id] = a.eqns[0].invars
    [lhs_l_id, rhs_l_id] = a.jaxpr.invars
    not_swap = lhs_g_id == lhs_l_id and rhs_g_id == rhs_l_id
    yes_swap = lhs_g_id == rhs_l_id and rhs_g_id == lhs_l_id
    assert not_swap != yes_swap

    cfg = copy.deepcopy(self.cfg)
    prng_name = self.prng_name

    lhs_quant_mode = self.lhs_quant_mode
    lhs_init = self.lhs_init
    lhs_scale_init = self.lhs_scale_init
    lhs_var_name = self.lhs_var_name

    rhs_quant_mode = self.rhs_quant_mode
    rhs_init = self.rhs_init
    rhs_scale_init = self.rhs_scale_init
    rhs_var_name = self.rhs_var_name

    quant_collection = self.quant_collection

    if yes_swap:
      if cfg is not None:
        cfg.fwd.lhs, cfg.fwd.rhs = cfg.fwd.rhs, cfg.fwd.lhs
        cfg.dlhs, cfg.drhs = cfg.drhs, cfg.dlhs
      lhs_quant_mode, rhs_quant_mode = rhs_quant_mode, lhs_quant_mode
      lhs_init, rhs_init = rhs_init, lhs_init
      lhs_scale_init, rhs_scale_init = rhs_scale_init, lhs_scale_init
      lhs_var_name, rhs_var_name = rhs_var_name, lhs_var_name

    aqt_dg = AqtDotGeneral(
        cfg=cfg,
        prng_name=prng_name,
        lhs_quant_mode=lhs_quant_mode,
        lhs_init=lhs_init,
        lhs_scale_init=lhs_scale_init,
        lhs_var_name=lhs_var_name,
        rhs_quant_mode=rhs_quant_mode,
        rhs_init=rhs_init,
        rhs_scale_init=rhs_scale_init,
        rhs_var_name=rhs_var_name,
        quant_collection=quant_collection,
        name=self.name,
    )
    return einsum(lhs_g, rhs_g, aqt_dg)


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
        use_fake_quant=False,
        # dtype_x=dtype,
        use_fwd_quant=None,
        preprocess=None,
        context=config.Context(key=None, train_step=None),
    )

  def dg_raw_config(lhs_bits, rhs_bits, local_aqt=None) -> config.DotGeneralRaw:
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
    )

  cfg = config.DotGeneral(
      fwd=dg_raw_config(fwd_bits, fwd_bits),
      dlhs=dg_raw_config(dlhs_bits, dlhs_bits, local_aqt=dlhs_local_aqt),
      drhs=dg_raw_config(drhs_bits, drhs_bits, local_aqt=drhs_local_aqt),
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

  return cfg
