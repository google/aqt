# Copyright 2023 Google LLC
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

"""Experimental transform to replace a primitive with a custom implementaion.

Example:

  from aqt.jax.aqt_custom_impl import custom_impl
  from jax import lax

  def f32_dot_general(x, y, **kwargs):
    x_f32 = x.astype('float32')
    y_f32 = y.astype('float32')
    return lax.dot_general(x_f32, y_f32, **kwargs).astype(x.dtype)

  @custom_impl(lax.dot_general_p, f32_dot_general)
  def func(x, y):
    return x @ y

  out = func(x, y)  # uses f32_dot_general in place of all dot_general calls
"""
import functools
from typing import Any, Callable, Dict, List, Sequence, TypeVar

import jax
from jax import api_util
from jax import core
from jax import lax
from jax import tree_util
from jax.experimental import pjit
from jax.interpreters import partial_eval as pe
from jax.linear_util import wrap_init


F = TypeVar('F', bound=Callable)
T = TypeVar('T')


def _safe_map(f: Callable[..., T], *args: Any) -> List[T]:
  args = list(map(list, args))
  if len(set(map(len, args))) != 1:
    raise ValueError(f'length mismatch: {list(map(len, args))}')
  return list(map(f, *args))


# This transformation is implemented in "initial style", following the
# approach outlined in the "Writing custom jaxpr interpreters in JAX" doc:
# https://jax.readthedocs.io/en/latest/notebooks/Writing_custom_interpreters_in_Jax.html


def custom_impl(
    prim: core.Primitive, impl: Callable[..., Any]
) -> Callable[[F], F]:
  """Experimental transformation to inject custom primitive implementations.

  Args:
    prim: JAX primitive
    impl: Function to use in place of this primitive

  Returns:
    transformation operator

  Example:

    from aqt.jax.aqt_custom_impl import custom_impl
    from jax import lax

    def f32_dot_general(x, y, **kwargs):
      x_f32 = x.astype('float32')
      y_f32 = y.astype('float32')
      return lax.dot_general(x_f32, y_f32, **kwargs).astype(x.dtype)

    @custom_impl(lax.dot_general_p, f32_dot_general)
    def func(x):
      return x.T @ x

    out = func(x)  # uses f32_dot_general in place of all dot_general calls
  """
  if not isinstance(prim, core.Primitive):
    raise ValueError(
        f'First argument to custom_impl should be a primitive. Got {prim}'
    )
  new_impls = {prim: impl}

  def custom_impl_transformation(fun):
    @functools.wraps(fun)
    def wrapped(*args, **kwargs):
      args_flat, in_tree = tree_util.tree_flatten((args, kwargs))
      in_avals_flat = [core.get_aval(arg) for arg in args_flat]
      wrapped_fun, out_tree = api_util.flatten_fun(wrap_init(fun), in_tree)
      jaxpr, out_avals_flat, consts = pe.trace_to_jaxpr_dynamic(
          wrapped_fun, in_avals_flat
      )
      result = _custom_impl_jaxpr(new_impls, jaxpr, consts, *args)
      assert len(out_avals_flat) == len(result)
      return tree_util.tree_unflatten(out_tree(), result)

    return wrapped

  return custom_impl_transformation


def _apply_custom_impl_to_jaxpr(
    closed_jaxpr: core.ClosedJaxpr,
    custom_impls: Dict[core.Primitive, Callable[..., Any]],
    *avals: core.ShapedArray,
) -> core.ClosedJaxpr:
  return jax.make_jaxpr(
      functools.partial(
          _custom_impl_jaxpr,
          custom_impls,
          closed_jaxpr.jaxpr,
          closed_jaxpr.literals,
      )
  )(*avals)


def _custom_impl_jaxpr(
    custom_impls: Dict[core.Primitive, Callable[..., Any]],
    jaxpr: core.Jaxpr,
    consts: Sequence[Any],
    *args: Any,
) -> Sequence[Any]:
  """Evaluate a jaxpr for the custom_impl transformation."""
  env = {}

  def read(var):
    if isinstance(var, core.Literal):
      return var.val
    return env[var]

  def write(var, val):
    env[var] = val

  _safe_map(write, jaxpr.invars, args)
  _safe_map(write, jaxpr.constvars, consts)

  for eqn in jaxpr.eqns:
    invals = _safe_map(read, eqn.invars)
    in_avals = [core.get_aval(inval) for inval in invals]
    # TODO(jakevdp): are there other higher-order primitives we need to support?
    if eqn.primitive in (pjit.pjit_p, lax.scan_p):
      new_jaxpr = _apply_custom_impl_to_jaxpr(
          eqn.params['jaxpr'], custom_impls, *in_avals
      )
      outvals = eqn.primitive.bind(
          *invals, **{**eqn.params, 'jaxpr': new_jaxpr}
      )
    elif eqn.primitive == lax.while_p:
      new_cond_jaxpr = _apply_custom_impl_to_jaxpr(
          eqn.params['cond_jaxpr'], custom_impls, *in_avals
      )
      new_body_jaxpr = _apply_custom_impl_to_jaxpr(
          eqn.params['body_jaxpr'], custom_impls, *in_avals
      )
      outvals = eqn.primitive.bind(
          *invals,
          **{
              **eqn.params,
              'cond_jaxpr': new_cond_jaxpr,
              'body_jaxpr': new_body_jaxpr,
          },
      )
    elif eqn.primitive in custom_impls:
      outvals = custom_impls[eqn.primitive](*invals, **eqn.params)
      out_avals = tree_util.tree_map(
          lambda val: api_util.shaped_abstractify(core.get_aval(val)), outvals
      )
      expected_out_avals = [var.aval for var in eqn.outvars]
      if not eqn.primitive.multiple_results:
        expected_out_avals = expected_out_avals[0]
      if out_avals != expected_out_avals:
        raise ValueError(
            f'custom impl for {eqn.primitive} returned the wrong output'
            f' types.\n  expected: {expected_out_avals}\n  actual:  '
            f' {out_avals}'
        )
    else:
      outvals = eqn.primitive.bind(*invals, **eqn.params)
    if not eqn.primitive.multiple_results:
      outvals = [outvals]
    _safe_map(write, eqn.outvars, outvals)
  return _safe_map(read, jaxpr.outvars)
