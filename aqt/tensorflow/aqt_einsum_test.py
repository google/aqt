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

"""Tests for einsum."""

import typing
from typing import Any, Dict, Optional, Sequence, Tuple
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from aqt.common import aqt_common
from aqt.common import aqt_config
from aqt.tensorflow import aqt_einsum
from aqt.tensorflow import aqt_ops
from aqt.tensorflow import aqt_tensor
from aqt.test import aqt_test_shared_base
import numpy as np
import tensorflow.compat.v1 as tf


def _stats_config(share_stats_axes: Sequence[int]) -> aqt_config.StatsConfig:
  """Generates dynamic quantization stats configuration."""
  return aqt_config.StatsConfig(
      ema_update_count=1,
      share_stats_axes=list(share_stats_axes),
      update_count_prior=0,
      tpu_cross_replica_sum=False)


def _schedule_config(
    bits: int, const_bound_coeff: float,
    share_stats_axes: Sequence[int]) -> aqt_config.AqtScheduleConfig:
  """Creates schedule config with dynamic quantization."""
  iqc = aqt_config.IntQuantConfig(bits=bits)
  cc = aqt_config.CalibrationConfig(const_bound_coeff=const_bound_coeff)
  tc = aqt_config.AqtTensorConfig(
      quant_config=iqc, calibration_config=cc, freeze_scale_at_begin=True)
  sc = _stats_config(share_stats_axes)
  return aqt_config.AqtScheduleConfig(sc, [tc])


def _schedule_config_emulation(
    share_stats_axes) -> aqt_config.AqtScheduleConfig:
  """Creates schedule config for emulated precision."""
  iqc = aqt_config.SmallFloatConfig(
      exponent_bits=5,
      mantissa_bits=2,
      min_exp=-14,
      max_exp=15,
      support_inf=False,
      rounding_mode=aqt_config.RoundingMode.ROUND_TO_NEAREST_EVEN)
  # Using the max number essentially disables scaling.
  cc = aqt_config.CalibrationConfig(
      const_bound_coeff=aqt_common._get_max_number_float(
          mantissa_bits=2, max_exp=15))
  tc = aqt_config.AqtTensorConfig(
      quant_config=iqc, calibration_config=cc, freeze_scale_at_begin=True)
  sc = aqt_config.StatsConfig(
      ema_update_count=1,
      share_stats_axes=list(share_stats_axes),
      update_count_prior=0,
      tpu_cross_replica_sum=False)
  return aqt_config.AqtScheduleConfig(sc, [tc])


def _empty_config(
    share_stats_axes: Sequence[int]) -> aqt_config.AqtScheduleConfig:
  return aqt_config.AqtScheduleConfig(_stats_config(share_stats_axes), [])


def _einsum_op(
    eq: str,  #
    lhs: tf.Tensor,
    rhs: tf.Tensor,
    lhs_config: aqt_config.AqtScheduleConfig,
    rhs_config: aqt_config.AqtScheduleConfig,
    lhs_weights: Optional[tf.Tensor] = None,
    rhs_weights: Optional[tf.Tensor] = None,
    varscope_name: str = "einsum",
    train: bool = True,
    quantize_bwd: bool = False,
    lhs_bwd_config: Optional[aqt_config.AqtScheduleConfig] = None,
    rhs_bwd_config: Optional[aqt_config.AqtScheduleConfig] = None,
    random_noise_seed: Optional[int] = 1234,
    **einsum_kwargs,
) -> tf.Tensor:
  """Updates quantizers at event_count=0 and computes einsum."""
  with tf.variable_scope(varscope_name):
    lhs_tq = aqt_tensor.TensorQuantizer(
        lhs.shape, lhs_config, name="lhs")
    rhs_tq = aqt_tensor.TensorQuantizer(
        rhs.shape, rhs_config, name="rhs")
    lhs_bwd_tq, rhs_bwd_tq = None, None
    grad_shape = aqt_einsum.get_out_shape(eq, lhs.shape, rhs.shape)
    if lhs_bwd_config:
      lhs_bwd_tq = aqt_tensor.TensorQuantizer(
          grad_shape, lhs_bwd_config, name="lhs_bwd"
      )
    if rhs_bwd_config:
      rhs_bwd_tq = aqt_tensor.TensorQuantizer(
          grad_shape, rhs_bwd_config, name="rhs_bwd"
      )
    if quantize_bwd and random_noise_seed is not None:
      random_gen = tf.random.Generator.from_seed(random_noise_seed)
    else:
      random_gen = None

  event_count = tf.constant(0, tf.int64)
  updates = [
      lhs_tq.update(lhs, lhs_weights, event_count),
      rhs_tq.update(rhs, rhs_weights, event_count)
  ]
  with tf.control_dependencies(updates):
    return aqt_ops.aqt_einsum(
        eq,
        lhs_tq,
        lhs,
        rhs_tq,
        rhs,
        train,
        quantize_bwd,
        lhs_bwd_tq,
        rhs_bwd_tq,
        random_gen=random_gen,
        **einsum_kwargs,
    )


def _generate_missing_shared_axes() -> Sequence[Dict[str, Any]]:
  """Cases where shared axes are missing."""

  keys = ["testcase_name", "eq", "lhs_share", "rhs_share"]
  cases: Sequence[Tuple[str, str, Sequence[int], Sequence[int]]] = [
      ("no_sharing_one_lhs_valid", "i,->i", [], []),
      ("no_sharing_one_rhs_valid", ",i->i", [], []),
      ("no_sharing_both_valid", "i,j->ij", [], []),
      ("contracting_partial_rhs", "j,j->", [], [0]),
      ("contracting_partial_lhs", "j,j->", [0], []),
      ("contracting_valid", "j,j->", [0], [0]),
  ]

  cases_dicts = []
  for vals in cases:
    case = dict(zip(keys, vals))
    case["is_valid"] = typing.cast(str, case["testcase_name"]).endswith("valid")
    cases_dicts.append(case)

  return cases_dicts


def _generate_diag_equation() -> Sequence[Dict[str, Any]]:
  """Cases where shared axes are missing."""

  keys = ["testcase_name", "eq", "lhs_rank", "rhs_rank"]
  cases = [
      ("diag_left", "ii,->i", 2, 0),
      ("diag_right", ",ii->i", 0, 2),
      ("diag_both", "ii,jj->ij", 2, 2),
      ("matmul_and_diag_left", "ii,ik->k", 2, 2),
      ("matmul_and_diag_right", "ij,jj->i", 2, 2),
  ]
  return [dict(zip(keys, vals)) for vals in cases]


def _generate_self_contracting_equation() -> Sequence[Dict[str, Any]]:
  """Cases where shared axes are missing."""

  keys = ["testcase_name", "eq", "lhs_rank", "rhs_rank"]
  cases = [
      ("sum_left", "i,->", 1, 0),
      ("sum_right", ",i->", 0, 1),
      ("sum_both", "i,j->", 1, 1),
      ("batch_matmul_sum_right", "bij,jk->bi", 3, 2),
      ("batch_matmul_sum_left", "bij,jk->bk", 3, 2),
      ("batch_matmul_sum_both", "bij,jk->b", 3, 2),
  ]
  return [dict(zip(keys, vals)) for vals in cases]


def _generate_bad_equation() -> Sequence[Dict[str, Any]]:
  """Cases where shared axes are missing."""

  keys = ["testcase_name", "eq", "lhs_rank", "rhs_rank"]
  cases = [
      ("single_arg", "ii->", 2, 0),
      ("single_arg_no_out", "ii", 2, 0),
      ("double_arg_no_out", "ii,ij", 2, 2),
      ("bad_out", "i,i>i", 2, 2),
      ("bad_out_dash", "i,i-i", 2, 2),
      ("nonstandard_axes", "i!j,ij->i", 3, 2),
      ("space", "i j,ij->i", 3, 2),
      ("newline", "ij,i\nj->i", 2, 3),
      ("ellipses", "...ij,jk->ik", 4, 2),
      ("diag", "ii,->i", 2, 0),
      ("sum", "i,->", 1, 0),
      ("trace", "ii,->", 2, 0),
  ]
  return [dict(zip(keys, vals)) for vals in cases]


def _generate_test_equations(
    append_quantize_bwd: bool = True) -> Sequence[Dict[str, Any]]:
  keys = ["testcase_name", "eq", "quantize_bwd"]
  eq_cases = [
      ("transpose", ",ij->ji"),
      ("matmul", "ij,jk->ik"),
      ("batch_matmul", "bij,bjk->bik"),
      ("dot", "i,i->"),
      ("vec_mat_mult", "u,uv->v"),
      ("channel_vec_mat_mult", "u,nuv->nv"),
      ("batch_channel_vec_mat_mult", "bu,nuv->bnv"),
      ("block_vec_mat_mult", "nu,nuv->nv"),
      ("batch_block_vec_mat_mult", "bnu,nuv->bnv"),
      ("batch_transpose_matmul", "bmu,nm->bnu"),
  ]

  if not append_quantize_bwd:
    return [dict(zip(keys, vals)) for vals in eq_cases]

  cases = []
  for name, eq in eq_cases:
    # append a value indicate whether quantize backward pass
    cases.append((name + "_quantize_bwd", eq, True))
    cases.append((name, eq, False))

  return [dict(zip(keys, vals)) for vals in cases]


def _generate_equations_with_axes() -> Sequence[Dict[str, Any]]:
  keys = ["testcase_name", "eq", "lhs_contracting_axes", "rhs_contracting_axes",
          "lhs_diagnal_axes", "rhs_diagnal_axes", "lhs_self_contracting_labels",
          "rhs_self_contracting_labels"]
  cases = [
      ("diag", "ii,->i", [], [], {"i": [0, 1]}, {}, [], []),
      ("sum", "i,->", [0], [], {}, {}, ["i"], []),
      ("trace", "ii,->", [0, 1], [], {"i": [0, 1]}, {}, ["i"], []),
      ("transpose", ",ij->ji", [], [], {}, {}, [], []),
      ("matmul", "ij,jk->ik", [1], [0], {}, {}, [], []),
      ("batch_matmul", "bij,bjk->bik", [2], [1], {}, {}, [], []),
      ("dot", "i,i->", [0], [0], {}, {}, [], []),
      ("vec_mat_mult", "u,uv->v", [0], [0], {}, {}, [], []),
      ("channel_vec_mat_mult", "u,nuv->nv", [0], [1], {}, {}, [], []),
      ("batch_channel_vec_mat_mult", "bu,nuv->bnv", [1], [1], {}, {}, [], []),
      ("block_vec_mat_mult", "nu,nuv->nv", [1], [1], {}, {}, [], []),
      ("batch_block_vec_mat_mult", "bnu,nuv->bnv", [2], [1], {}, {}, [], []),
      ("batch_transpose_matmul", "bmu,nm->bnu", [1], [1], {}, {}, [], []),
  ]
  return [dict(zip(keys, vals)) for vals in cases]


class EinsumTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    """Seed random for deterministic but nontrivial inputs."""
    super().setUp()
    self.rng = np.random.default_rng(1234)

  def randn(self, *size):
    return self.rng.standard_normal(size=size, dtype=np.float32)

  @parameterized.named_parameters(_generate_missing_shared_axes())
  def test_missing_shared_axes(
      self,  #
      eq: str,
      lhs_share: Sequence[int],
      rhs_share: Sequence[int],
      is_valid: bool):

    def make_tensor(einsum_str):
      return tf.constant(np.ones([1] * len(einsum_str)), tf.float32)

    def make_op():
      lhs, rhs, _ = aqt_einsum._parse_equation(eq)
      lhs, rhs = make_tensor(lhs), make_tensor(rhs)
      return _einsum_op(eq, lhs, rhs, _empty_config(lhs_share),
                        _empty_config(rhs_share))

    if is_valid:
      make_op()
    else:
      with self.assertRaisesRegex(aqt_config.ConfigError,
                                  "axis .* of .* must be shared due to .*"):
        make_op()

  @parameterized.named_parameters(_generate_diag_equation())
  def test_diag_equation(self, eq: str, lhs_rank: int, rhs_rank: int):
    with self.assertRaisesRegex(aqt_config.ConfigError,
                                "expected to have no diagnalization"):

      def make_tensor(rank):
        return tf.constant(np.ones([1] * rank), tf.float32)

      lhs = make_tensor(lhs_rank)
      rhs = make_tensor(rhs_rank)
      lhs_config = _empty_config(list(range(lhs_rank)))
      rhs_config = _empty_config(list(range(rhs_rank)))
      _einsum_op(eq, lhs, rhs, lhs_config, rhs_config)

  @parameterized.named_parameters(_generate_self_contracting_equation())
  def test_self_contracting_equation(self, eq: str, lhs_rank: int,
                                     rhs_rank: int):
    with self.assertRaisesRegex(aqt_config.ConfigError,
                                "expected to have no self-contraction"):

      def make_tensor(rank):
        return tf.constant(np.ones([1] * rank), tf.float32)

      lhs = make_tensor(lhs_rank)
      rhs = make_tensor(rhs_rank)
      lhs_config = _empty_config(list(range(lhs_rank)))
      rhs_config = _empty_config(list(range(rhs_rank)))
      _einsum_op(eq, lhs, rhs, lhs_config, rhs_config)

  @parameterized.named_parameters(_generate_bad_equation())
  def test_bad_equation(self, eq: str, lhs_rank: int, rhs_rank: int):
    with self.assertRaisesRegex(aqt_config.ConfigError, "einsum equation"):

      def make_tensor(rank):
        return tf.constant(np.ones([1] * rank), tf.float32)

      lhs = make_tensor(lhs_rank)
      rhs = make_tensor(rhs_rank)
      lhs_config = _empty_config(list(range(lhs_rank)))
      rhs_config = _empty_config(list(range(rhs_rank)))
      _einsum_op(eq, lhs, rhs, lhs_config, rhs_config)

  def basic_quant_example(self):
    eq = "bji,jk->bik"

    # Representable values: -6, -4, -2, 0, 2, 4, 6 (preserving zeros).
    # Data batch axis: 0
    # Contracting axis: 1
    lhs_bound = 7
    lhs_config = _schedule_config(3, lhs_bound, [0, 1])
    lhs = tf.constant(
        np.array(
            [
                [
                    [-8, -5.99],  #
                    [4.01, 0.01],
                    [4.01, -4.01]
                ],
                [
                    [-0.01, 2.01],  #
                    [4.01, 6.01],
                    [3.99, -3.99]
                ]
            ],
            dtype=np.float32))
    qlhs_value = tf.constant(
        np.array(
            [
                [
                    [-6, -6],  #
                    [4, 0],
                    [4, -4]
                ],
                [
                    [0, 2],  #
                    [4, 6],
                    [4, -4]
                ]
            ],
            dtype=np.float32))

    # manual clip and quantization to be differentiable.
    qlhs = tf.clip_by_value(lhs, -lhs_bound, lhs_bound)
    qlhs = tf.grad_pass_through(lambda x: qlhs_value)(qlhs)

    # Representable values: -1, 0, 1
    # Contracting axis: 0
    rhs_bound = 1.5
    rhs_config = _schedule_config(2, 1.5, [0])
    rhs = tf.constant(
        np.array(
            [
                [-3, 0.99],  #
                [-0.99, 0],
                [-0.01, 2]
            ],
            dtype=np.float32))
    qrhs_value = tf.constant(
        np.array(
            [
                [-1, 1],  #
                [-1, 0],
                [0, 1]
            ],
            dtype=np.float32))

    # similar manual clip and quantization but with a different clip bound
    qrhs = tf.clip_by_value(rhs, -rhs_bound, rhs_bound)
    qrhs = tf.grad_pass_through(lambda x: qrhs_value)(qrhs)

    return eq, lhs_config, lhs, qlhs, rhs_config, rhs, qrhs

  def basic_emulation_example(self):
    eq = "ij,jk->ik"
    # Data batch axis: 0
    # Contracting axis: 1
    lhs_config = _schedule_config_emulation([0, 1])
    lhs = tf.constant(
        np.array(
            [
                [-8.5, 4.3, 4.1],  #
                [-0.05, 0.01, -4.7],
            ],
            dtype=np.float32))
    qlhs = tf.constant(
        np.array(
            [
                [-8.0, 4.0, 4.0],  #
                [-0.046875, 0.00976562, -5.0]
            ],
            dtype=np.float32))

    # Contracting axis: 0
    rhs_config = _schedule_config_emulation([0])
    rhs = tf.constant(
        np.array(
            [
                [-0.2, 0.02],  #
                [-1.1, 0],
                [-0.04, 2.3]
            ],
            dtype=np.float32))
    qrhs = tf.constant(
        np.array(
            [
                [-0.1875, 0.01953125],  #
                [-1.0, 0.0],
                [-0.0390625, 2.5]
            ],
            dtype=np.float32))

    return eq, lhs_config, lhs, qlhs, rhs_config, rhs, qrhs

  def test_basic_einsum(self):
    with self.subTest("quant_example"):
      eq, lhs_config, lhs, qlhs, rhs_config, rhs, qrhs = (
          self.basic_quant_example())

    with self.subTest("emulation_example"):
      eq, lhs_config, lhs, qlhs, rhs_config, rhs, qrhs = (
          self.basic_emulation_example())

    actual = _einsum_op(eq, lhs, rhs, lhs_config, rhs_config)
    expected = tf.einsum(eq, qlhs, qrhs)

    with self.cached_session() as sess, sess.as_default():
      tf.global_variables_initializer().run()
      self.assertAllEqual(expected, actual)

  def test_no_quantization(self):
    lhs = tf.constant(self.randn(3, 4))
    rhs = tf.constant(self.randn(4, 2))
    eq = "ij,jk->ik"
    lhs_config = _empty_config([1])
    rhs_config = _empty_config([0])

    lhs_float_config = _schedule_config(8, 1.0, [1])
    rhs_float_config = _schedule_config(8, 1.0, [0])

    lhs_float_config.tensor_configs[0].quant_config = aqt_config.FloatConfig()
    rhs_float_config.tensor_configs[0].quant_config = aqt_config.FloatConfig()

    actual_fwd = _einsum_op(
        eq, lhs, rhs, lhs_config, rhs_config, varscope_name="no_quant_einsum")
    float_config_actual_fwd = _einsum_op(eq, lhs, rhs, lhs_float_config,
                                         rhs_float_config,
                                         varscope_name="float_config_einsum")
    expected_fwd = tf.einsum(eq, lhs, rhs)

    actual_lgrad, actual_rgrad = tf.gradients([actual_fwd], [lhs, rhs])
    float_config_actual_lgrad, float_config_actual_rgrad = tf.gradients(
        [float_config_actual_fwd], [lhs, rhs])
    expected_lgrad, expected_rgrad = tf.gradients([expected_fwd], [lhs, rhs])

    with self.cached_session() as sess, sess.as_default():
      tf.global_variables_initializer().run()
      with self.subTest("fwd_int"):
        self.assertAllEqual(expected_fwd, actual_fwd)
      with self.subTest("fwd_float"):
        self.assertAllEqual(expected_fwd, float_config_actual_fwd)
      with self.subTest("bwd_int"):
        self.assertAllEqual(actual_lgrad, expected_lgrad)
        self.assertAllEqual(actual_rgrad, expected_rgrad)
      with self.subTest("bwd_float"):
        self.assertAllEqual(float_config_actual_lgrad, expected_lgrad)
        self.assertAllEqual(float_config_actual_rgrad, expected_rgrad)

  @parameterized.parameters([dict(lhs_float=True), dict(lhs_float=False)])
  def test_float_config_basic_einsum(self, lhs_float):
    eq, lhs_config, lhs, qlhs, rhs_config, rhs, qrhs = self.basic_quant_example(
    )
    if lhs_float:
      lhs_config.tensor_configs[0].quant_config = aqt_config.FloatConfig()
    else:
      rhs_config.tensor_configs[0].quant_config = aqt_config.FloatConfig()

    actual_fwd = _einsum_op(eq, lhs, rhs, lhs_config, rhs_config)
    if lhs_float:
      qlhs = lhs  # lhs is not quantized
    else:
      qrhs = rhs  # rhs is not quantized
    expected_fwd = tf.einsum(eq, qlhs, qrhs)

    actual_lgrad, actual_rgrad = tf.gradients([actual_fwd], [lhs, rhs])
    expected_lgrad, expected_rgrad = tf.gradients([expected_fwd], [lhs, rhs])

    with self.cached_session() as sess, sess.as_default():
      tf.global_variables_initializer().run()
      with self.subTest("fwd"):
        self.assertAllEqual(expected_fwd, actual_fwd)
      with self.subTest("bwd"):
        self.assertAllEqual(actual_lgrad, expected_lgrad)
        self.assertAllEqual(actual_rgrad, expected_rgrad)

  def test_passes_arguments_to_inner_einsum(self):
    module = "tensorflow.compat.v1"
    with mock.patch(f"{module}.einsum", side_effect=tf.einsum) as tfeinsum:
      lhs = tf.constant(self.randn(3, 4))
      rhs = tf.constant(self.randn(4, 2))
      eq = "ij,jk->ik"
      lhs_config = _schedule_config(8, 1.0, [1])
      rhs_config = _schedule_config(8, 1.0, [0])

      kwargs = {"optimize": "optimal", "name": "optimal_einsum"}

      _einsum_op(eq, lhs, rhs, lhs_config, rhs_config, **kwargs)
      for (_, actual_kwargs) in tfeinsum.call_args_list:
        subset = {k: v for k, v in actual_kwargs.items() if k in kwargs}
        self.assertEqual(subset, kwargs)

  @parameterized.named_parameters(
      aqt_test_shared_base.generate_unaligned_schedule_intervals())
  def test_unaligned_schedule_intervals(self, lhs_intervals, rhs_intervals):

    def config_from_schedule(intervals,
                             share_stats_axes):
      config = _empty_config(share_stats_axes)
      for start, stop in intervals:
        config.tensor_configs += _schedule_config(
            8, 1.0, share_stats_axes).tensor_configs
        config.tensor_configs[-1].begin_at_event = start
        config.tensor_configs[-1].end_at_event = stop
      return config

    with self.assertRaisesRegex(aqt_config.ConfigError,
                                "intervals do not match|config len"):
      lhs = tf.constant(self.randn(3, 4))
      rhs = tf.constant(self.randn(4, 2))
      eq = "ij,jk->ik"
      # share_stats_axes includes the batch axies i and the contracting axis j.
      lhs_config = config_from_schedule(lhs_intervals, [0, 1])
      rhs_config = config_from_schedule(rhs_intervals, [0])
      _einsum_op(eq, lhs, rhs, lhs_config, rhs_config)

  def exact_int8_einsum_example(
      self,
      eq: str,
      quantize_lhs: bool = False,
      quantize_rhs: bool = False,
      scale: float = 1.0,
      quantize_bwd: bool = False,
      dynamic_bwd_quant: bool = False,
  ):
    """Returns a pair of tensors and config to einsum exactly."""
    lhs, rhs, _ = aqt_einsum._parse_equation(eq)

    # A subset of the range of numbers which can be preserved exactly.
    bits = 8
    symmetric_uniform_range = 2**(bits - 1) - 1
    lo, hi = -symmetric_uniform_range, symmetric_uniform_range

    axis_labels = sorted(set(lhs + rhs))
    label_dims = {k: self.rng.integers(2, 10) for k in axis_labels}
    lhs_shape = [label_dims[k] for k in lhs]
    rhs_shape = [label_dims[k] for k in rhs]

    def make_tensor(shape):
      np_tensor = self.rng.integers(lo, hi, size=shape, dtype=np.int64)
      return tf.constant(np_tensor, dtype=tf.float32)

    lhs = make_tensor(lhs_shape) * scale
    rhs = make_tensor(rhs_shape) * scale

    def _exact_schedule_config(bits, eq, scale):
      iqc = aqt_config.IntQuantConfig(bits=bits, preserve_zero=True)
      clip_bound = aqt_common.get_clip_bound(iqc)
      assert symmetric_uniform_range <= clip_bound

      lhs_caxes, rhs_caxes = aqt_einsum.get_contracting_axes(eq)

      # to exactly represent quantized lhs and rhs
      const_bound_coeff = scale * clip_bound
      lhs_config = _schedule_config(bits, const_bound_coeff, lhs_caxes)
      rhs_config = _schedule_config(bits, const_bound_coeff, rhs_caxes)
      return lhs_config, rhs_config

    lhs_config, rhs_config = _exact_schedule_config(8, eq, scale)

    lhs_config.use_quantized_variable = quantize_lhs
    rhs_config.use_quantized_variable = quantize_rhs

    def _get_grad_config(eq: str,
                         swap_ans: bool
                         ) -> Optional[aqt_config.AqtScheduleConfig]:
      if not quantize_bwd:
        return None
      bwd_eq = aqt_einsum.get_einsum_transpose(eq, swap_ans=swap_ans)
      # 16 bits to preserve gradients
      grad_config, _ = _exact_schedule_config(16, bwd_eq, 1.0)
      return grad_config

    lhs_bwd_config = _get_grad_config(eq, False)
    rhs_bwd_config = _get_grad_config(eq, True)

    # Change calibration configs if dynamic quant in the backward pass
    if lhs_bwd_config and dynamic_bwd_quant:
      for tc in lhs_bwd_config.tensor_configs:
        tc.calibration_config = aqt_config.CalibrationConfig(
            const_bound_coeff=1.0,
            max_dev_coeff=1.0,
        )
    if rhs_bwd_config and dynamic_bwd_quant:
      for tc in rhs_bwd_config.tensor_configs:
        tc.calibration_config = aqt_config.CalibrationConfig(
            const_bound_coeff=1.0,
            max_dev_coeff=1.0,
        )

    return lhs_config, lhs, rhs_config, rhs, lhs_bwd_config, rhs_bwd_config

  @parameterized.named_parameters(_generate_test_equations())
  def test_vars_dont_kill_grads(self, eq, quantize_bwd):
    with self.subTest("scale_one"):
      lhs_config, lhs, rhs_config, rhs, lbwd_config, rbwd_config = (
          self.exact_int8_einsum_example(
              eq, True, True, scale=1.0, quantize_bwd=quantize_bwd,
          )
      )
      lhs_config_novar, _, rhs_config_novar, _, lbwd_config, rbwd_config = (
          self.exact_int8_einsum_example(
              eq, True, True, scale=1.0, quantize_bwd=quantize_bwd
          )
      )
    with self.subTest("scale_two"):
      lhs_config, lhs, rhs_config, rhs, lbwd_config, rbwd_config = (
          self.exact_int8_einsum_example(
              eq, True, True, scale=2.0, quantize_bwd=quantize_bwd
          )
      )
      lhs_config_novar, _, rhs_config_novar, _, lbwd_config, rbwd_config = (
          self.exact_int8_einsum_example(
              eq, True, True, scale=2.0, quantize_bwd=quantize_bwd
          )
      )

    expected_op = _einsum_op(
        eq,
        lhs,
        rhs,
        lhs_config_novar,
        rhs_config_novar,
        varscope_name="novar",
        quantize_bwd=quantize_bwd,
        lhs_bwd_config=lbwd_config,
        rhs_bwd_config=rbwd_config,
    )
    actual_op = _einsum_op(
        eq,
        lhs,
        rhs,
        lhs_config,
        rhs_config,
        varscope_name="var",
        quantize_bwd=quantize_bwd,
        lhs_bwd_config=lbwd_config,
        rhs_bwd_config=rbwd_config,
    )

    saved_grads = tf.gradients([expected_op], [lhs, rhs])
    unsaved_grads = tf.gradients([actual_op], [lhs, rhs])

    with self.cached_session() as sess, sess.as_default():
      tf.global_variables_initializer().run()

      zipped_grads = zip(saved_grads, unsaved_grads)
      for actual_grad, expected_grad in zipped_grads:
        self.assertAllEqual(actual_grad, expected_grad)

  @parameterized.named_parameters(_generate_test_equations())
  def test_vars_over_inputs_at_inference(self, eq, quantize_bwd):
    with self.subTest("scale_one"):
      lhs_config, lhs, rhs_config, rhs, lhs_bwd_config, rhs_bwd_config = (
          self.exact_int8_einsum_example(
              eq, True, True, scale=1.0, quantize_bwd=quantize_bwd
          )
      )
    with self.subTest("scale_two"):
      lhs_config, lhs, rhs_config, rhs, lhs_bwd_config, rhs_bwd_config = (
          self.exact_int8_einsum_example(
              eq, True, True, scale=2.0, quantize_bwd=quantize_bwd
          )
      )

    lhs_tq = aqt_tensor.TensorQuantizer(lhs.shape, lhs_config, name="lhs")
    rhs_tq = aqt_tensor.TensorQuantizer(rhs.shape, rhs_config, name="rhs")
    if quantize_bwd:
      grad_shape = aqt_einsum.get_out_shape(eq, lhs.shape, rhs.shape)
      lhs_bwd_tq = aqt_tensor.TensorQuantizer(
          grad_shape, lhs_bwd_config, name="lhs_bwd"
      )
      rhs_bwd_tq = aqt_tensor.TensorQuantizer(
          grad_shape, rhs_bwd_config, name="rhs_bwd"
      )
      random_gen = tf.random.Generator.from_seed(1234)
    else:
      lhs_bwd_tq = rhs_bwd_tq = None
      random_gen = None

    # Update at least once to initialize scale, then grab the expected
    # value while in training mode.
    event_count = tf.constant(0, tf.int64)
    updates = [
        lhs_tq.update(lhs, weight=None, event_count=event_count),
        rhs_tq.update(rhs, weight=None, event_count=event_count)
    ]
    with tf.control_dependencies(updates):
      expected = aqt_ops.aqt_einsum(
          eq,
          lhs_tq,
          lhs,
          rhs_tq,
          rhs,
          train=True,
          quantize_bwd=quantize_bwd,
          lhs_grad_quantizer=lhs_bwd_tq,
          rhs_grad_quantizer=rhs_bwd_tq,
          random_gen=random_gen,
      )

    with self.cached_session() as sess, sess.as_default():
      tf.global_variables_initializer().run()
      expected = expected.eval()

      actual = aqt_ops.aqt_einsum(
          eq,
          lhs_tq,
          tf.zeros_like(lhs),
          rhs_tq,
          tf.zeros_like(rhs),
          train=False,
          quantize_bwd=quantize_bwd,
          lhs_grad_quantizer=lhs_bwd_tq,
          rhs_grad_quantizer=rhs_bwd_tq,
          random_gen=random_gen,
      )

      self.assertAllEqual(actual, expected)

  @parameterized.named_parameters(_generate_test_equations())
  def test_float_config_not_save_quantized_var(self, eq, quantize_bwd):
    with self.subTest("scale_one"):
      lhs_config, lhs, rhs_config, rhs, lhs_bwd_config, rhs_bwd_config = (
          self.exact_int8_einsum_example(
              eq, True, True, scale=1.0, quantize_bwd=quantize_bwd
          )
      )
    with self.subTest("scale_two"):
      lhs_config, lhs, rhs_config, rhs, lhs_bwd_config, rhs_bwd_config = (
          self.exact_int8_einsum_example(
              eq, True, True, scale=2.0, quantize_bwd=quantize_bwd
          )
      )

    lhs_config.tensor_configs[0].quant_config = aqt_config.FloatConfig()
    lhs_tq = aqt_tensor.TensorQuantizer(lhs.shape, lhs_config, name="lhs")
    rhs_tq = aqt_tensor.TensorQuantizer(rhs.shape, rhs_config, name="rhs")
    if quantize_bwd:
      grad_shape = aqt_einsum.get_out_shape(eq, lhs.shape, rhs.shape)
      lhs_bwd_tq = aqt_tensor.TensorQuantizer(
          grad_shape, lhs_bwd_config, name="lhs_bwd"
      )
      rhs_bwd_tq = aqt_tensor.TensorQuantizer(
          grad_shape, rhs_bwd_config, name="rhs_bwd"
      )
      random_gen = tf.random.Generator.from_seed(1234)
    else:
      lhs_bwd_tq = rhs_bwd_tq = None
      random_gen = None

    event_count = tf.constant(0, tf.int64)

    with self.cached_session() as sess, sess.as_default():
      tf.global_variables_initializer().run()
      lhs_tq.update(lhs, weight=None, event_count=event_count).run()
      rhs_tq.update(rhs, weight=None, event_count=event_count).run()

      actual = aqt_ops.aqt_einsum(
          eq,
          lhs_tq,
          lhs,
          rhs_tq,
          rhs,
          train=False,
          quantize_bwd=quantize_bwd,
          lhs_grad_quantizer=lhs_bwd_tq,
          rhs_grad_quantizer=rhs_bwd_tq,
          random_gen=random_gen,
      )
      # Although the input tensors are non-zeros, the result of einsum with
      # inference mode should be zeros because lhs uses zero-initialized
      # quantized var while rhs can restore its updated quantized variable.
      expected = tf.zeros_like(actual)

      self.assertAllEqual(actual, expected)

  @parameterized.named_parameters(_generate_test_equations())
  def test_exact(self, eq, quantize_bwd):
    with self.subTest("scale_one"):
      lhs_config, lhs, rhs_config, rhs, lhs_bwd_config, rhs_bwd_config = (
          self.exact_int8_einsum_example(
              eq, True, True, scale=1.0, quantize_bwd=quantize_bwd
          )
      )
    with self.subTest("scale_two"):
      lhs_config, lhs, rhs_config, rhs, lhs_bwd_config, rhs_bwd_config = (
          self.exact_int8_einsum_example(
              eq, True, True, scale=2.0, quantize_bwd=quantize_bwd
          )
      )

    actual = _einsum_op(
        eq,
        lhs,
        rhs,
        lhs_config,
        rhs_config,
        quantize_bwd=quantize_bwd,
        lhs_bwd_config=lhs_bwd_config,
        rhs_bwd_config=rhs_bwd_config,
    )
    expected = tf.einsum(eq, lhs, rhs)

    with self.cached_session() as sess, sess.as_default():
      tf.global_variables_initializer().run()
      self.assertAllEqual(actual, expected)

  @parameterized.named_parameters(_generate_test_equations())
  def test_exact_grads(self, eq, quantize_bwd):
    with self.subTest("scale_one"):
      lhs_config, lhs, rhs_config, rhs, lhs_bwd_config, rhs_bwd_config = (
          self.exact_int8_einsum_example(
              eq, True, True, scale=1.0, quantize_bwd=quantize_bwd,
          )
      )
    with self.subTest("scale_two"):
      lhs_config, lhs, rhs_config, rhs, lhs_bwd_config, rhs_bwd_config = (
          self.exact_int8_einsum_example(
              eq, True, True, scale=2.0, quantize_bwd=quantize_bwd,
          )
      )

    random_noise_seed = 1234 if quantize_bwd else None
    actual_fwd = _einsum_op(
        eq,
        lhs,
        rhs,
        lhs_config,
        rhs_config,
        quantize_bwd=quantize_bwd,
        lhs_bwd_config=lhs_bwd_config,
        rhs_bwd_config=rhs_bwd_config,
        random_noise_seed=random_noise_seed,
    )
    expected_fwd = tf.einsum(eq, lhs, rhs)

    expected = tf.gradients([expected_fwd], [lhs, rhs])
    actual = tf.gradients([actual_fwd], [lhs, rhs])

    with self.cached_session() as sess, sess.as_default():
      tf.global_variables_initializer().run()
      for actual_grad, expected_grad in zip(actual, expected):
        # Uniform noises in [-5, 5] while {-5, 5} has zero measure.
        # Expect the gradient is still exact since (-5, 5) does not change
        # rouding.
        self.assertAllEqual(actual_grad, expected_grad)

  @parameterized.named_parameters(_generate_test_equations())
  def test_inexact(self, eq, quantize_bwd):
    with self.subTest("larger_x_bound"):
      lhs_config, lhs, rhs_config, rhs, lhs_bwd_config, rhs_bwd_config = (
          self.exact_int8_einsum_example(
              eq, scale=2.0, quantize_bwd=quantize_bwd
              )
      )
      lhs, rhs = lhs / 2.0, rhs / 2.0
    with self.subTest("smaller_x_bound"):
      lhs_config, lhs, rhs_config, rhs, lhs_bwd_config, rhs_bwd_config = (
          self.exact_int8_einsum_example(
              eq, scale=1.0, quantize_bwd=quantize_bwd
              )
      )
      lhs, rhs = lhs * 2.0, rhs * 2.0

    actual = _einsum_op(
        eq,
        lhs,
        rhs,
        lhs_config,
        rhs_config,
        quantize_bwd=quantize_bwd,
        lhs_bwd_config=lhs_bwd_config,
        rhs_bwd_config=rhs_bwd_config,
    )
    expected = tf.einsum(eq, lhs, rhs)

    with self.cached_session() as sess, sess.as_default():
      tf.global_variables_initializer().run()
      self.assertNotAllEqual(actual, expected)

  @parameterized.named_parameters(_generate_test_equations())
  def test_inexact_grads(self, eq, quantize_bwd):
    with self.subTest("larger_x_bound"):
      lhs_config, lhs, rhs_config, rhs, lhs_bwd_config, rhs_bwd_config = (
          self.exact_int8_einsum_example(
              eq, scale=2.0, quantize_bwd=quantize_bwd
          )
      )
      lhs, rhs = lhs / 2.0, rhs / 2.0
    with self.subTest("smaller_x_bound"):
      lhs_config, lhs, rhs_config, rhs, lhs_bwd_config, rhs_bwd_config = (
          self.exact_int8_einsum_example(
              eq, scale=1.0, quantize_bwd=quantize_bwd
          )
      )
      lhs, rhs = lhs * 2.0, rhs * 2.0

    actual_fwd = _einsum_op(
        eq,
        lhs,
        rhs,
        lhs_config,
        rhs_config,
        quantize_bwd=quantize_bwd,
        lhs_bwd_config=lhs_bwd_config,
        rhs_bwd_config=rhs_bwd_config,
    )
    expected_fwd = tf.einsum(eq, lhs, rhs)

    expected = tf.gradients([expected_fwd], [lhs, rhs])
    actual = tf.gradients([actual_fwd], [lhs, rhs])

    with self.cached_session() as sess, sess.as_default():
      tf.global_variables_initializer().run()
      for actual_grad, expected_grad in zip(actual, expected):
        self.assertNotAllEqual(actual_grad, expected_grad)

  @parameterized.named_parameters(_generate_test_equations(
      # Testing bwd quant only, no need to include quantize_bwd in test cases.
      append_quantize_bwd=False,
      ))
  def test_consistent_bwd_improves_grads(self, eq):
    lhs_config, lhs, rhs_config, rhs, lhs_bwd_config, rhs_bwd_config = (
        self.exact_int8_einsum_example(
            eq, quantize_bwd=True, dynamic_bwd_quant=True,
        )
    )
    def get_perturbed_gradients(random_noise_seed):
      actual_fwd = _einsum_op(
          eq,
          lhs,
          rhs,
          lhs_config,
          rhs_config,
          quantize_bwd=True,
          lhs_bwd_config=lhs_bwd_config,
          rhs_bwd_config=rhs_bwd_config,
          random_noise_seed=random_noise_seed,
          varscope_name=f"einsum_seed_{random_noise_seed}",
      )
      return tf.gradients([actual_fwd], [lhs, rhs])

    exact_fwd = tf.einsum(eq, lhs, rhs)
    exact = tf.gradients([exact_fwd], [lhs, rhs])

    biased = get_perturbed_gradients(None)
    biased_errors = [tf.linalg.norm(i - j) for i, j in zip(biased, exact)]

    num_samples = 8
    qgrad_samples = [get_perturbed_gradients(i) for i in range(num_samples)]
    estimate1 = qgrad_samples[0]
    estimate8 = [tf.reduce_mean(g, axis=0) for g in zip(*qgrad_samples[:8])]

    def get_error(estimate):
      return [tf.linalg.norm(i - j) for i, j in zip(estimate, exact)]

    sample_errors = get_error(estimate1)
    ensemble_errors = get_error(estimate8)

    with self.cached_session() as sess, sess.as_default():
      tf.global_variables_initializer().run()

      for biased_g, exact_g, sample_error, ensemble_err, biased_err in zip(
          biased, exact, sample_errors, ensemble_errors, biased_errors
          ):
        # Check dynamic backward quant is inexact
        self.assertNotAllEqual(biased_g, exact_g)
        # unbiased estimate should have smaller errors than the biased one
        self.assertAllLess(ensemble_err, biased_err)
        # the unbiased estimate should eventually converge or make improvement
        self.assertAllLess(ensemble_err, sample_error)

  @parameterized.named_parameters(_generate_equations_with_axes())
  def test_equations_with_axes(self, eq,
                               lhs_contracting_axes, rhs_contracting_axes,
                               lhs_diagnal_axes, rhs_diagnal_axes,
                               lhs_self_contracting_labels,
                               rhs_self_contracting_labels):
    l_axes, r_axes = aqt_einsum.get_contracting_axes(eq)
    for actual, expected in [(l_axes, lhs_contracting_axes),
                             (r_axes, rhs_contracting_axes)]:
      self.assertAllEqual(actual, expected)
    lhs, rhs, out = aqt_einsum._parse_equation(eq)
    for actual, expected in [(aqt_einsum._get_diagnal_axes(lhs),
                              lhs_diagnal_axes),
                             (aqt_einsum._get_diagnal_axes(rhs),
                              rhs_diagnal_axes)]:
      self.assertAllEqual(actual, expected)
    lc, rc = aqt_einsum._get_self_contracting_labels(lhs, rhs, out)
    for actual, expected in [(lc, lhs_self_contracting_labels),
                             (rc, rhs_self_contracting_labels)]:
      self.assertAllEqual(actual, expected)

if __name__ == "__main__":
  absltest.main()
