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
"""Quantized einsum.

Quantized einsum for the analogous :py:func:`tf.einsum`.

Note that only 2-argument einsum is supported. For details, see go/aqtp-einsum.
In a two argument einsum, we have an equation `lhs,rhs->out` where `lhs`, `rhs`,
and `out` are strings containing single-character axes labels. Note that a
tensor axis is a distinct notion from einsum axes labels (henceforth, "labels"),
since a tensor axis is a natural number indexing from 0 to one less than that
tensor's dimension, but labels can be repeated within a single tensor.

We thus define an important notion for quantization of einsum: contracting axes,
which are tensor axes whose labels are not present in the output but in both
arguments.

We do not support self-contraction (e.g., sum (i,->)), which requires to know
its shape when transpose in the backward pass. We also do not support
diagonalization (e.g., ii,->i). For these two cases, consider using an
one-argument einsum beforehand.

For instance, in matrix multiply (ij,jk->ik), axes with label j are contracting.

Of course, if an axis is contracting, all axes across all inputs with the same
label are contracting.
"""

import collections
import string
from typing import Dict, Iterable, List, Optional, Tuple

from aqt.common import aqt_config
from aqt.common import aqt_config_utils
from aqt.tensorflow import aqt_ops_util
from aqt.tensorflow import aqt_tensor
import tensorflow.compat.v1 as tf

# We repeatedly use protected methods from classes defined in other modules to
# avoid exporting them as part of the public API.
# pylint: disable=protected-access


def _parse_equation(eq: str) -> Tuple[str, str, str]:
  """Parses a two-argument einsum equation.

  Args:
    eq: the two-argument einsum equation.

  Returns:
    A tuple `(lhs, rhs, out)` for the separate parts of the einsum equation.

  Raises:
    aqt_config.ConfigError: `eq` is not strictly in the form `lhs,rhs->out`
    where
    `lhs`, `rhs`, and `out` are strings of characters `[a-zA-Z]`. Thus, ellipses
    and whitspace in equations is not supported.
  """
  args, sep, out = eq.partition('->')
  if not sep:
    raise aqt_config.ConfigError(
        f'einsum equation ("{eq}") expected to have in/out delimiter "->"')

  num_commas = args.count(',')
  if num_commas != 1:
    raise aqt_config.ConfigError(
        f'einsum equation ("{eq}") expected to have 2 arguments, but '
        f'{num_commas+1} found (consider splitting your expression into '
        'multiple two-argument einsums, see go/aqtp-einsum)')

  lhs, sep, rhs = args.partition(',')
  assert sep

  for labels in [lhs, rhs, out]:
    for char in labels:
      if char not in string.ascii_letters:
        raise aqt_config.ConfigError(
            f'einsum equation ("{eq}") has illegal character "{char}"')

  return lhs, rhs, out


def _validate_shared_axes(
    lhs_config: aqt_config.StatsConfig,  # Prevent python auto-formatting.
    rhs_config: aqt_config.StatsConfig | None,
    eq: str,
) -> None:
  """Validates that contracting axes have shared statistics."""
  lc_axes, rc_axes = get_contracting_axes(eq)
  lb_axes, rb_axes = get_batch_axes(eq)

  axes_name_configs = [(lc_axes, lb_axes, 'lhs', lhs_config),
                       (rc_axes, rb_axes, 'rhs', rhs_config)]
  for c_axes, b_axes, name, config in axes_name_configs:
    # Skip if no stats config provided, used to verify the backward pass.
    if not config:
      continue
    shared_indices = set(config.share_stats_axes)
    for i in c_axes:
      if i not in shared_indices:
        raise aqt_config.ConfigError(
            f'axis {i} of {name} must be shared due to contraction')
    remaining_share_axes = shared_indices - set(c_axes) - set(b_axes)
    if remaining_share_axes:
      raise aqt_config.ConfigError(
          f'axes ("{remaining_share_axes}") of {name} must be either batch '
          f'axes {b_axes} or contracting axes {c_axes}')


def _validate_equation(eq: str) -> None:
  """Validates arguments in the two-argument einsum equation.

  We do not support a einsum equation that includes:
    diagonalization: repeated labels in a argument, e.g., 'ii,->i'.
    sum (or self-contraction): contraction only in one argument, e.g., 'i,->'.
  in either of the two arguments. As a result, this also excludes trace
  ('ii,->'). Please consider to use a one-argument einsum beforehand.

  Args:
    eq: The einsum equation.

  Raises:
    aqt_config.ConfigError: if the equation includes diagonalization or
    self-contraction in either of the two arguments.
  """
  lhs, rhs, out = _parse_equation(eq)
  lhs_diag_axes = _get_diagnal_axes(lhs)
  rhs_diag_axes = _get_diagnal_axes(rhs)
  for diag_axes, name, axes in [(lhs_diag_axes, 'lhs', lhs),
                                (rhs_diag_axes, 'rhs', rhs)]:
    if diag_axes:
      diag_labels = ', '.join(diag_axes.keys())
      raise aqt_config.ConfigError(f'einsum equation ("{eq}") expected to have '
                                   f'no diagnalization but {name} ("{axes}") '
                                   f'has diagnalization in: "{diag_labels}".'
                                   'Consider using a one-argument einsum '
                                   'beforehand for the diagnalization case.')
  lhs_sc, rhs_sc = _get_self_contracting_labels(lhs, rhs, out)
  for sc, name, axes in [(lhs_sc, 'lhs', lhs), (rhs_sc, 'rhs', rhs)]:
    if sc:
      sc_labels = ', '.join(sc)
      raise aqt_config.ConfigError(f'einsum equation ("{eq}") expected to have '
                                   f'no self-contraction but {name} ("{axes}") '
                                   f'has self-contraction in: "{sc_labels}".'
                                   'Consider using a one-argument einsum '
                                   'beforehand for the self-contraction case.')


def get_contracting_axes(
    eq: str) -> Tuple[List[Optional[int]], List[Optional[int]]]:
  """Returns contracting axes in the einsum equation.

  Contracting axes are defined as axes in lhs and rhs.

  Args:
    eq: einsum equation

  Returns:
    A tuple of lhs and rhs contracting axes.
  """
  lhs, rhs, out = _parse_equation(eq)
  contracting_labels = set(lhs + rhs) - set(out)

  lhs_contracting_axes, rhs_contracting_axes = [], []
  for labels, axes in (lhs, lhs_contracting_axes), (rhs, rhs_contracting_axes):
    for i, label in enumerate(labels):
      if label in contracting_labels:
        axes.append(i)

  return lhs_contracting_axes, rhs_contracting_axes


def get_batch_axes(eq: str) -> Tuple[List[Optional[int]], List[Optional[int]]]:
  """Returns batch axes in the einsum equation.

  Batch axes are defined as axes in out and one of lhs and rhs. Batch axes are
  possibly data batch axes since data batch can only appear in lhs or rhs but
  not both.

  Args:
    eq: einsum equation

  Returns:
    A tuple of batch axes for lhs and rhs
  """
  lhs, rhs, out = _parse_equation(eq)
  # batch labels that appear in out and one of lhs and rhs
  lb_labels = list(set(out).intersection(set(lhs)) - set(rhs))
  rb_labels = list(set(out).intersection(set(rhs)) - set(lhs))
  lb_axes, rb_axes = [], []
  for labels, axes, b_labels in [(lhs, lb_axes, lb_labels),
                                 (rhs, rb_axes, rb_labels)]:
    for i, label in enumerate(labels):
      if label in b_labels:
        axes.append(i)

  return lb_axes, rb_axes


def get_out_shape(
    eq: str, lhs_shape: Iterable[int], rhs_shape: Iterable[int]
) -> List[int]:
  labeled_dimensions = {}
  lhs, rhs, out = _parse_equation(eq)
  for axes, shape in [(lhs, lhs_shape), (rhs, rhs_shape)]:
    for label, dim in zip(axes, shape):
      if label in labeled_dimensions:
        assert labeled_dimensions[label] == dim
      else:
        labeled_dimensions[label] = dim
  return [labeled_dimensions[label] for label in out]


def default_einsum(
    eq: str,  #
    lhs_quantizer: aqt_tensor.TensorQuantizer | None,
    rhs_quantizer: aqt_tensor.TensorQuantizer,
    lhs: tf.Tensor,
    rhs: tf.Tensor,
    train: bool,
    **tf_einsum_kwargs,
) -> tf.Tensor:
  """Perform tf.einsum with input tensors of float type.

  The gradient quantizer (assumed to be lhs) in the backward pass is optional.

  Args:
    eq: einsum equation
    lhs_quantizer: TensorQuantizer for lhs
    rhs_quantizer: TensorQuantizer for rhs
    lhs: lhs of einsum
    rhs: rhs of einsum
    train: If false and `use_quantized_variable` in lhs_quantizer or
      rhs_quantizer, then this indicates `aqt_einsum` should use the quantized
      variable with the latest quantized, memorized from the most recent
      `TensorQuantizer.update()` in quantized operations rather than the float
      tensor input `lhs` or `rhs` provided to those operations at inference
      time.
    **tf_einsum_kwargs: Keyword arguments to pass onto `einsum`.

  Returns:
    result of einsum
  """
  if lhs_quantizer:
    lhs = aqt_ops_util._possibly_use_quantized_variable(
        lhs_quantizer, lhs, train
    )
  rhs = aqt_ops_util._possibly_use_quantized_variable(rhs_quantizer, rhs, train)
  return tf.einsum(eq, lhs, rhs, **tf_einsum_kwargs)


def _scale_grad(grad: tf.Tensor, y_inv_scale: tf.Tensor, grad_dims: str,
                y_dims: str, y_share_stats_axes: Iterable[int],
                **tf_einsum_kwargs,
                ) -> tf.Tensor:
  # Remove share stats dimensions of size 1, which are not appear in out
  sy_inv_scale = tf.squeeze(y_inv_scale, axis=y_share_stats_axes)
  sy_axes = [d for i, d in enumerate(y_dims) if i not in y_share_stats_axes]
  sy_dims = ''.join(sy_axes)
  # Use einsum to scale the gradients instead of using broadcasting.
  scale_eq = f'{grad_dims},{sy_dims}->{grad_dims}'
  return tf.einsum(scale_eq, grad, sy_inv_scale, **tf_einsum_kwargs)


def _get_diagnal_axes(x_dims: str) -> Dict[str, int]:
  """Returns repeated labels which indicates diagonalization."""
  diagonal_axes = collections.defaultdict(list)
  for i, d in enumerate(x_dims):
    if x_dims.count(d) > 1:
      diagonal_axes[d].append(i)
  return diagonal_axes


def _get_self_contracting_labels(lhs: str, rhs: str, out: str
                                 ) -> Tuple[List[str], List[str]]:
  """Returns labels that appears in lhs or rhs only.

  Given a two-argument einsum equation "lhs,rhs->out", a self-contracting label
  is a label that only appears in lhs or rhs but not in any two of lhs, rhs and
  out. For example, "i,->" contains contraction in "i" which only appears in the
  lhs.

  Args:
    lhs: the first argument in the einsum equation
    rhs: the second argument in the einsum equation
    out: the output of the einsum equation

  Returns:
    A tuple of self-contracting labels in lhs and rhs
  """
  return (
      list(set(lhs) - set(rhs) - set(out)),
      list(set(rhs) - set(lhs) - set(out)),
  )


def get_einsum_transpose(eq: str, swap_ans: bool = False) -> str:
  """Returns Einsum transpose equation used for a backward pass.

  Assume equation is two-argument: x,y->out.

  Args:
    eq: einsum equation.
    swap_ans: If not swap_ans, returns out,y->x; If swap_ans, returns out,x->y.

  Returns:
    Transpose equation.
  """
  if swap_ans:
    y_dims, x_dims, out_dims = _parse_equation(eq)
  else:
    x_dims, y_dims, out_dims = _parse_equation(eq)
  return '{},{}->{}'.format(out_dims, y_dims, x_dims)


def _maybe_random(
    random_gen: Optional[tf.random.Generator],
    shape: Iterable[int],
    dtype: tf.dtypes.DType,
) -> Optional[tf.Tensor]:
  """Maybe generate random floats in [-0.5, 0.5] to perturb gradients."""
  if random_gen is None:
    return None
  return random_gen.uniform(shape, -0.5, 0.5, dtype=dtype)


def _round(
    x_quantizer: aqt_tensor.TensorQuantizer,
    x: tf.Tensor,
    random: Optional[tf.Tensor],
    train: bool,
) -> tf.Tensor:
  if random is None:
    return x_quantizer._to_quant(x, train=train)
  assert x.shape == random.shape, (x.shape, random.shape)
  return x_quantizer._to_quant(x + random, train=train)


def einsum(
    eq: str,  #
    lhs_quantizer: aqt_tensor.TensorQuantizer,
    lhs: tf.Tensor,
    rhs_quantizer: aqt_tensor.TensorQuantizer,
    rhs: tf.Tensor,
    train: bool = True,
    quantize_bwd: bool = False,
    lhs_grad_quantizer: Optional[aqt_tensor.TensorQuantizer] = None,
    rhs_grad_quantizer: Optional[aqt_tensor.TensorQuantizer] = None,
    random_gen: Optional[tf.random.Generator] = None,
    **tf_einsum_kwargs,
) -> tf.Tensor:
  """Performs a quantized two-argument :py:func:`tf.einsum`.

  Args:
    eq: The einsum equation.
    lhs_quantizer: TensorQuantizer for lhs
    lhs: A `tf.Tensor`. Must be `float32`. The convolution lhs.
    rhs_quantizer: TensorQuantizer for rhs
    rhs: A `tf.Tensor`. Must have the same type as `lhs`. The convolution
      kernel.
    train: If false and `use_quantized_variable` in lhs_quantizer or
      rhs_quantizer, then this indicates `aqt_einsum` should use the quantized
      variable with the latest quantized, memorized from the most recent
      `TensorQuantizer.update()` in quantized operations rather than the float
      tensor input `lhs` or `rhs` provided to those operations at inference
      time.
    quantize_bwd: Whether to quantize the backward pass. If true, both
      lhs_grad_quantizer and rhs_grad_quantizer have to be conformal to dynamic
      quantization (only `const_bound_coeff` and `max_dev_coeff` are non-zero).
    lhs_grad_quantizer: A `TensorQuantizer` for grad, which is used to quantize
      the einsum equation, `grad,rhs->lhs_grad`, in the backward pass.
    rhs_grad_quantizer: A `TensorQuantizer` for grad, which is used to quantize
      the einsum equation, `grad,lhs->rhs_grad`, in the backward pass.
    random_gen: A `tf.random.Generator` used to generate random numbers between
      [0.5, 0.5] added to the gradients before quantization in the backward
      pass.
    **tf_einsum_kwargs: Keyword arguments to pass onto `einsum`.

  Returns:
    A `float32` tensor conformal to what `tf.einsum` would return.

  Raises:
    aqt_config.ConfigError: if the equation is not a two-argument einsum,
    the quantization schedules between arguments are misaligned, or a
    contracting axis does not have shared statistics.
  """
  _validate_equation(eq)
  _validate_shared_axes(
      lhs_quantizer.config.stats_config,  #
      rhs_quantizer.config.stats_config,
      eq)
  aqt_config_utils._validate_alignment(
      'lhs_quantizer.config.tensor_configs',  #
      lhs_quantizer.config.tensor_configs,
      'rhs_quantizer.config.tensor_configs',
      rhs_quantizer.config.tensor_configs)

  if not quantize_bwd:
    assert lhs_grad_quantizer is None
    assert rhs_grad_quantizer is None
  else:
    assert lhs_grad_quantizer is not None
    assert rhs_grad_quantizer is not None
    lhs_bwd_eq = get_einsum_transpose(eq, swap_ans=False)
    _validate_equation(lhs_bwd_eq)
    _validate_shared_axes(
        lhs_grad_quantizer.config.stats_config,  #
        None,  # only verify gradient quantizer in the backward pass
        lhs_bwd_eq,
    )
    aqt_config_utils._validate_alignment(
        'lhs_grad_quantizer.config.tensor_configs',  #
        lhs_grad_quantizer.config.tensor_configs,
        'rhs_quantizer.config.tensor_configs',
        rhs_quantizer.config.tensor_configs,
    )
    # validate 'grad,lhs->rhs_grad'
    rhs_bwd_eq = get_einsum_transpose(eq, swap_ans=True)
    _validate_equation(rhs_bwd_eq)
    _validate_shared_axes(
        rhs_grad_quantizer.config.stats_config,  #
        None,  # only verify gradient quantizer in the backward pass
        rhs_bwd_eq,
    )
    aqt_config_utils._validate_alignment(
        'rhs_grad_quantizer.config.tensor_configs',  #
        rhs_grad_quantizer.config.tensor_configs,
        'lhs_quantizer.config.tensor_configs',
        lhs_quantizer.config.tensor_configs,
    )

    def _is_dynamic_calibration(config: aqt_config.CalibrationConfig) -> bool:
      return config.l1_dev_coeff == 0.0 and config.lp_dev_coeff == 0.0

    for name, tensor_configs in [
        (
            'lhs_grad_calibration_config',
            lhs_grad_quantizer.config.tensor_configs,
        ),
        (
            'rhs_grad_calibration_config',
            rhs_grad_quantizer.config.tensor_configs,
        ),
    ]:
      for tensor_config in tensor_configs:
        cali_config = tensor_config.calibration_config
        if not _is_dynamic_calibration(cali_config):
          raise ValueError(
              'The backward-pass quantization assumes dynamic quant while the '
              f'calibration config for {name} has l1_dev_coeff = '
              f'{cali_config.l1_dev_coeff} and lp_dev_coeff = '
              f'{cali_config.lp_dev_coeff}, both of which should be zero.'
          )

  def fwd(lhs: tf.Tensor,
          rhs: tf.Tensor) -> tf.Tensor:
    with tf.name_scope('AqtEinsum'):
      with tf.name_scope('get_quant_scale'):
        with tf.name_scope('lhs'):
          lhs_scale, lhs_inv_scale = lhs_quantizer._get_quant_scale(train)
        with tf.name_scope('rhs'):
          rhs_scale, rhs_inv_scale = rhs_quantizer._get_quant_scale(train)

      lhs_scaled = lhs_scale * lhs
      rhs_scaled = rhs_scale * rhs

      with tf.name_scope('to_quant'):
        with tf.name_scope('lhs'):
          qlhs = lhs_quantizer._to_quant(lhs_scaled, train)
        with tf.name_scope('rhs'):
          qrhs = rhs_quantizer._to_quant(rhs_scaled, train)

      # TODO(vladf): until tf.einsum supports int8 arguments, we need to cast
      # the quantized variables to a floating point format.
      with tf.name_scope('einsum'):
        out = default_einsum(eq, lhs_quantizer, rhs_quantizer, qlhs, qrhs,
                             train, **tf_einsum_kwargs)

      with tf.name_scope('inv_scale'):
        assert len(lhs_inv_scale.shape) == len(qlhs.shape)
        assert len(rhs_inv_scale.shape) == len(qrhs.shape)
        inv_scale = tf.einsum(eq, lhs_inv_scale, rhs_inv_scale,
                              **tf_einsum_kwargs)
      return out * inv_scale

  @tf.custom_gradient
  def qeinsum(lhs: tf.Tensor,
              rhs: tf.Tensor) -> tf.Tensor:

    out = fwd(lhs, rhs)

    def bwd(grad: tf.Tensor) -> tf.Tensor:
      # Make sure to build all backprop results after fprop is computed.
      # Since we rely on variables being updated, this is important for
      # consistency. For instance, the forward pass might be computed under
      # a user-added control dependency from lhs and rhs update; the
      # backward pass should also share this dependency transitively.
      with tf.control_dependencies([out]):
        # Note the differences between autograd and what we
        # have implemented below:
        #
        # (1) We re-quantize lhs, rhs to save memory between
        #     fprop and bprop, this is manual rematerialization.
        # (2) Each bprop matmul can avoid a multiplication
        #     by an scale and inverse scale of an fprop argument
        #     due to cancellation.
        with tf.name_scope('BwdAqtEinsum'):

          with tf.name_scope('get_quant_scale'):
            with tf.name_scope('lhs'):
              lhs_scale, lhs_inv_scale = lhs_quantizer._get_quant_scale(train)
            with tf.name_scope('rhs'):
              rhs_scale, rhs_inv_scale = rhs_quantizer._get_quant_scale(train)

          lhs_scaled = lhs_scale * lhs
          rhs_scaled = rhs_scale * rhs

          def _bwd(
              eq: str,
              grad_quantizer: Optional[aqt_tensor.TensorQuantizer],
              y_quantizer: aqt_tensor.TensorQuantizer,
              grad: tf.Tensor,
              qy: tf.Tensor,
              y_inv_scale: tf.Tensor,
              train: bool,
          ) -> tf.Tensor:
            grad_dims, y_dims, _ = _parse_equation(eq)
            y_share_stats_axes = (
                y_quantizer.config.stats_config.share_stats_axes
            )
            grad = _scale_grad(
                grad,
                y_inv_scale,
                grad_dims,
                y_dims,
                y_share_stats_axes,
                **tf_einsum_kwargs,
            )
            if grad_quantizer:
              with tf.name_scope('to_quant_grad'):
                # We assume the backward-pass quantization is dynamic so no need
                # to pass weight when updating stats but still need _last_update
                # to switch tensor configs.
                update = grad_quantizer.update(
                    grad,
                    weight=None,
                    event_count=lhs_quantizer._last_update,
                )
                with tf.control_dependencies([update]):
                  grad_scale, grad_inv_scale = grad_quantizer._get_quant_scale(
                      train
                  )
                  grad_scaled = grad_scale * grad
                  random = _maybe_random(random_gen, grad.shape, grad.dtype)
                  qgrad = _round(grad_quantizer, grad_scaled, random, train)
                  assert len(grad_inv_scale.shape) == len(qgrad.shape)
            else:
              qgrad = grad
              grad_inv_scale = None

            with tf.name_scope('einsum'):
              out = default_einsum(
                  eq,
                  grad_quantizer,
                  y_quantizer,
                  qgrad,
                  qy,
                  train,
                  **tf_einsum_kwargs,
              )

            with tf.name_scope('inv_scale'):
              if grad_quantizer:
                grad_dims, _, x_dims = _parse_equation(eq)
                grad_share_stats_axes = (
                    grad_quantizer.config.stats_config.share_stats_axes
                )
                out = _scale_grad(
                    out,
                    grad_inv_scale,
                    x_dims,
                    grad_dims,
                    grad_share_stats_axes,
                    **tf_einsum_kwargs,
                )
            return out

          with tf.name_scope('lhs'):
            qrhs = rhs_quantizer._to_quant(rhs_scaled, train)
            lhs_transpose_eq = get_einsum_transpose(eq, swap_ans=False)
            lhs_bwd = _bwd(
                lhs_transpose_eq,
                lhs_grad_quantizer,
                rhs_quantizer,
                grad,
                qrhs,
                rhs_inv_scale,
                train,
            )
            lhs_bwd = tf.where_v2(
                lhs_quantizer._clip_mask(lhs_scaled, train), 0.0, lhs_bwd)

          with tf.name_scope('rhs'):
            qlhs = lhs_quantizer._to_quant(lhs_scaled, train)
            rhs_transpose_eq = get_einsum_transpose(eq, swap_ans=True)
            rhs_bwd = _bwd(
                rhs_transpose_eq,
                rhs_grad_quantizer,
                lhs_quantizer,
                grad,
                qlhs,
                lhs_inv_scale,
                train,
            )
            rhs_bwd = tf.where_v2(
                rhs_quantizer._clip_mask(rhs_scaled, train), 0.0, rhs_bwd)

        return [lhs_bwd, rhs_bwd]

    return out, bwd

  # If not training, do not install the custom gradient since it would cause
  # both sets of weights (float + int8) to be pulled into the graph, making it
  # difficult to prune away the unused set of weights for serving.
  if train:
    return qeinsum(lhs, rhs)
  else:
    return fwd(lhs, rhs)
