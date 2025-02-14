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
"""dot_general with flax lifted custom_vjp."""
from aqt.jax.v2 import aqt_dot_general
from aqt.jax.v2 import aqt_tensor
from aqt.jax.v2 import utils
import flax.linen as nn
import jax
from jax import numpy as jnp


def dg_core_flax_lifted(lhs: jnp.ndarray, rhs: jnp.ndarray,
    lhs_qt: None | aqt_tensor.QTensor, rhs_qt: None | aqt_tensor.QTensor,
    dimension_numbers: jax.lax.DotDimensionNumbers, owg_variables: utils.OWGVariables,
    mdl: nn.Module, cfg: aqt_dot_general.DotGeneral):
  """dot_general with flax lifted custom_vjp applied on it.

  Args:
    lhs: Left hand side argument of dot_general.
    rhs: Right hand side argument of dot_general.
    lhs_qt: Left hand side QTensor, if applicable.
    rhs_qt: Right hand side QTensor, if applicable.
    dimension_numbers: Dimension numbers for dot_general.
    mdl: Flax module in which the dot_general is called.
    cfg: aqt DotGeneral.

  Returns:
    aqt DotGeneral result. Flax-lifted custom_vjp is applied on it.
    :param owg_variables:
  """

  def _dg_core_flax_lifted(
      mdl: nn.Module,
      lhs: jnp.ndarray,
      rhs: jnp.ndarray,
      lhs_qt: None | aqt_tensor.QTensor,
      rhs_qt: None | aqt_tensor.QTensor,
      dimension_numbers: jax.lax.DotDimensionNumbers,
      owg_variables: utils.OWGVariables,
      cfg: aqt_dot_general.DotGeneral,
  ):
    """Lifted dg_core."""
    # The order of parameters must match with that of jax version to avoid the
    # backward gradient - primal order mismatch.
    del mdl
    return cfg.dg_core(lhs=lhs, rhs=rhs, lhs_qt=lhs_qt, rhs_qt=rhs_qt,
                      dimension_numbers=dimension_numbers, owg_variables=owg_variables)

  def _dg_core_vjp_fwd_flax_lifted(
      mdl: nn.Module,
      lhs: jnp.ndarray,
      rhs: jnp.ndarray,
      lhs_qt: None | aqt_tensor.QTensor,
      rhs_qt: None | aqt_tensor.QTensor,
      dimension_numbers: jax.lax.DotDimensionNumbers,
      owg_variables: utils.OWGVariables,
      cfg: aqt_dot_general.DotGeneral,
  ):
    """Lifted custom vjp_fwd."""

    # Currently we do not support backward computation for the variables
    # declared INSIDE the AqtDotGeneral layer.
    # Since we do not have any gradient functions for the params, removing
    # jax.lax.stop_gradient here will lead to NotImplementedError of
    # differentiation rules for 'custom_lin'. MutableArrays should not be passed
    # through stop_gradient, so we need to filter those out from this logic.
    def stop_grad_for_non_ref_params(param):
      if isinstance(
          jax.core.get_aval(param),
          # pylint: disable-next=protected-access
          jax._src.state.types.AbstractRef,
      ):
        return param
      return jax.lax.stop_gradient(param)

    params = jax.tree_util.tree_map(stop_grad_for_non_ref_params, mdl.variables)
    out, res = aqt_dot_general.dg_core_vjp_fwd(
        lhs, rhs, lhs_qt, rhs_qt, dimension_numbers, owg_variables, cfg
    )
    return out, (res, params)

  def _dg_core_vjp_bwd_flax_lifted(
      fwd_dimension_numbers: jax.lax.DotDimensionNumbers,
      res: tuple[
          tuple[
              None | aqt_dot_general.DotGeneralRes,
              aqt_dot_general.DotGeneral,
              None | utils.OWGVariables
          ],
          jax.core.ParamDict,
      ],
      g,
  ):
    """Lifted custom vjp_bwd."""
    res, params = res
    tangents = aqt_dot_general.dg_core_vjp_bwd(fwd_dimension_numbers, res, g)

    return params, *tangents

  dg_core_with_custom_vjp = nn.custom_vjp(
      _dg_core_flax_lifted,
      _dg_core_vjp_fwd_flax_lifted,
      _dg_core_vjp_bwd_flax_lifted,
      # Having a variable collection which does not included in the fwd/
      # bwd pipeline would trigger a complex pytype tree matching issue.
      grad_vars=True,
      nondiff_argnums=(5,),
  )
  return dg_core_with_custom_vjp(
      mdl, lhs, rhs, lhs_qt, rhs_qt, dimension_numbers, owg_variables, cfg
  )
