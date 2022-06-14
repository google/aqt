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

"""Tests for matmul."""

import copy
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from aqt.common import aqt_config
from aqt.tensorflow import aqt_matmul
from aqt.tensorflow import aqt_tensor
from aqt.test import aqt_test_shared_base
import numpy as np
import tensorflow.compat.v1 as tf


def test_stats_config():
  """Config base for all the experiments."""
  return aqt_config.StatsConfig(
      ema_update_count=1,  # single update overwrites the stats
      update_count_prior=0,  # easier equations
      share_stats_axes=[0, 1],  # one stat per whole tensor
      tpu_cross_replica_sum=False,  # no cross-tpu reduction
      filter_zeros=False,  # on default zeros are a number like any other
  )


def i64(x):
  """Convenience tf.i64 wrapper."""
  return tf.constant(x, dtype=tf.int64)


def calibration_config(const_coeff: float) -> aqt_config.CalibrationConfig:
  return aqt_config.CalibrationConfig(const_bound_coeff=const_coeff)


def _schedule_config(bits, const_bound_coeff,
                     share_stats_axes) -> aqt_config.AqtScheduleConfig:
  """Creates schedule config with dynamic quantization."""
  iqc = aqt_config.IntQuantConfig(bits=bits)
  cc = aqt_config.CalibrationConfig(const_bound_coeff=const_bound_coeff)
  tc = aqt_config.AqtTensorConfig(
      quant_config=iqc, calibration_config=cc, freeze_scale_at_begin=True)
  sc = aqt_config.StatsConfig(
      ema_update_count=1,
      share_stats_axes=list(share_stats_axes),
      update_count_prior=0,
      tpu_cross_replica_sum=False)
  return aqt_config.AqtScheduleConfig(sc, [tc])


def config_from_schedule(schedule):
  """Generates a schedule config from [(start, end, bits), ...]."""
  tensor_configs = []
  for start, end, bits in schedule:
    int_quant_config = aqt_config.IntQuantConfig(bits, preserve_zero=False)

    tensor_config = aqt_config.AqtTensorConfig(
        freeze_scale_at_begin=True,
        quant_config=int_quant_config,
        begin_at_event=start,
        end_at_event=end,
        calibration_config=calibration_config(1.0))

    tensor_configs.append(tensor_config)
  return aqt_config.AqtScheduleConfig(test_stats_config(), tensor_configs)


def update_event_count(matmul, event_count_int: int):
  """Update the quantizer's event count without changing stats."""
  for quantizer in [matmul.lhs_quantizer, matmul.rhs_quantizer]:
    sample = tf.zeros(quantizer.data_shape)
    weights = tf.zeros([1] * len(quantizer.data_shape))
    event_count = tf.constant(event_count_int, tf.int64)
    quantizer.update(sample, weights, event_count).run()


def matmul_config(matmul):
  """Creates an AqtMatmulConfig corresponding to a Matmul."""
  return aqt_config.AqtMatmulConfig(matmul.lhs_quantizer.config,
                                    matmul.rhs_quantizer.config)


class IntNarrowedMatMulTest(tf.test.TestCase, parameterized.TestCase):

  def test_chooses_right_matmul(self):
    # Create a list of settings (left_bits, right_bits, expected_matmul)
    # and generate a schedule based on those settings.
    settings = [(8, 16, "default"), (4, 16, "default"), (4, 8, "int8"),
                (8, 9, "default"), (4, 4, "int8"), (3, 3, "int8"),
                (9, 7, "default")]

    lhs_schedule = []
    rhs_schedule = []
    expected_results = []
    for i, (l, r, expected) in enumerate(settings):
      lhs_schedule.append((i, i + 1, l))
      rhs_schedule.append((i, i + 1, r))
      expected_results.append(expected)

    lhs_config = config_from_schedule(lhs_schedule)
    rhs_config = config_from_schedule(rhs_schedule)

    shape = [1, 1]  # Any shape will do, we're mocking.
    lhs_quant = aqt_tensor.TensorQuantizer(shape, lhs_config, name="lhs")
    rhs_quant = aqt_tensor.TensorQuantizer(shape, rhs_config, name="rhs")

    module = "aqt.tensorflow.aqt_matmul"
    with mock.patch(f"{module}.default_matmul") as default_matmul, \
         mock.patch(f"{module}.int8_matmul") as int8_matmul:
      default_matmul.return_value = tf.constant("default")
      int8_matmul.return_value = tf.constant("int8")

      event_ph = tf.placeholder(tf.int64)
      lhs_quant._last_update = event_ph
      rhs_quant._last_update = event_ph
      tf_actual = aqt_matmul._matmul_case(lhs_quant, rhs_quant, None, None,
                                          True)

      with self.cached_session():
        tf.global_variables_initializer().run()
        for i, expected in enumerate(expected_results):
          actual = tf_actual.eval(feed_dict={event_ph: i})
          self.assertEqual(
              actual.decode("utf-8"), expected, msg=f"event_count {i}")


class MatmulTest(tf.test.TestCase, parameterized.TestCase):

  def exact_int8_matmul_example(self,
                                lhs_use_quantized_variable=False,
                                rhs_use_quantized_variable=False):
    lhs_config, lhs, rhs_config, rhs = aqt_test_shared_base.exact_int8_example(
        lhs_shape=(3, 2),
        rhs_shape=(2, 2),
        lhs_share_stats_axes=[0, 1],
        rhs_share_stats_axes=[0, 1],
        lhs_use_quantized_variable=lhs_use_quantized_variable,
        rhs_use_quantized_variable=rhs_use_quantized_variable)

    lhs = tf.constant(lhs, tf.float32)
    rhs = tf.constant(rhs, tf.float32)

    config = aqt_config.AqtMatmulConfig(lhs_config, rhs_config)
    mm = aqt_matmul.Matmul(config, lhs.shape, rhs.shape)

    return mm, lhs, rhs

  # No quantization and float config give aqt_matmul = tf.matmul
  def test_matmul_none(self):
    no_quant_config = aqt_config.AqtScheduleConfig(test_stats_config(), [])
    float_config_tc = aqt_config.AqtTensorConfig(
        freeze_scale_at_begin=True,
        quant_config=aqt_config.FloatConfig(),
        calibration_config=calibration_config(1))
    float_config = aqt_config.AqtScheduleConfig(test_stats_config(),
                                                [float_config_tc])

    lhs = np.random.uniform(-1.0, 1.0, size=(2, 3)).astype(np.float32)
    lhs = tf.constant(lhs)

    rhs = np.random.uniform(-1.0, 1.0, size=(3, 4)).astype(np.float32)
    rhs = tf.constant(rhs)

    config = aqt_config.AqtMatmulConfig(no_quant_config, no_quant_config)
    mm_no_quant = aqt_matmul.Matmul(config, lhs.shape, rhs.shape, "no_quant")
    config = aqt_config.AqtMatmulConfig(float_config, float_config)
    mm_float = aqt_matmul.Matmul(config, lhs.shape, rhs.shape, "float")

    no_quant_ret = mm_no_quant.apply(lhs, rhs)
    float_config_ret = mm_float.apply(lhs, rhs)
    expected_ret = tf.matmul(lhs, rhs)

    with self.cached_session():
      tf.global_variables_initializer().run()
      self.assertAllEqual(no_quant_ret, expected_ret)
      self.assertAllEqual(float_config_ret, expected_ret)

  def basic_quant_example(self):
    lhs_config = _schedule_config(3, 7, [0, 1])
    lhs = tf.constant(
        np.array(
            [
                [-8, 4.01, 4.01],  #
                [-5.99, 0.01, -4.01],
            ],
            dtype=np.float32))
    qlhs = tf.constant(
        np.array(
            [
                [-6, 4, 4],  #
                [-6, 0, -4]
            ],
            dtype=np.float32))

    # Representable values: -1, 0, 1
    rhs_config = _schedule_config(2, 1.5, [0, 1])
    rhs = tf.constant(
        np.array(
            [
                [-3, 0.99],  #
                [-0.99, 0],
                [-0.01, 2]
            ],
            dtype=np.float32))
    qrhs = tf.constant(
        np.array(
            [
                [-1, 1],  #
                [-1, 0],
                [0, 1]
            ],
            dtype=np.float32))

    config = aqt_config.AqtMatmulConfig(lhs_config, rhs_config)

    return config, lhs, qlhs, rhs, qrhs

  def test_basic_matmul(self):
    config, lhs, qlhs, rhs, qrhs = self.basic_quant_example()

    mm = aqt_matmul.Matmul(config, lhs.shape, rhs.shape)

    event_count = tf.constant(0, tf.int64)
    updates = [
        mm.update_lhs(lhs, None, event_count),
        mm.update_rhs(rhs, None, event_count)
    ]
    with tf.control_dependencies(updates):
      actual = mm.apply(lhs, rhs)
    expected = tf.matmul(qlhs, qrhs)

    with self.cached_session() as sess, sess.as_default():
      tf.global_variables_initializer().run()
      self.assertAllEqual(expected, actual)

  @parameterized.parameters([dict(lhs_float=True), dict(lhs_float=False)])
  def test_float_config_basic_matmul(self, lhs_float):
    config, lhs, qlhs, rhs, qrhs = self.basic_quant_example()

    if lhs_float:
      config.lhs.tensor_configs[0].quant_config = aqt_config.FloatConfig()
    else:
      config.rhs.tensor_configs[0].quant_config = aqt_config.FloatConfig()

    mm = aqt_matmul.Matmul(config, lhs.shape, rhs.shape)

    event_count = tf.constant(0, tf.int64)
    updates = [
        mm.update_lhs(lhs, None, event_count),
        mm.update_rhs(rhs, None, event_count)
    ]
    with tf.control_dependencies(updates):
      actual = mm.apply(lhs, rhs)

    if lhs_float:
      qlhs = lhs  # lhs is not quantized
    else:
      qrhs = rhs  # rhs is not quantized
    expected = tf.matmul(qlhs, qrhs)

    with self.cached_session() as sess, sess.as_default():
      tf.global_variables_initializer().run()
      self.assertAllEqual(expected, actual)

  def with_config(self, mm, config):
    """Returns new Matmul with the new config but otherwise the same."""
    with tf.variable_scope(None, default_name="uniqued"):
      return aqt_matmul.Matmul(config, mm.lhs_quantizer.data_shape,
                               mm.rhs_quantizer.data_shape, mm.name,
                               mm.lhs_name, mm.rhs_name)

  def test_validates_contraction(self):
    mm, _, _ = self.exact_int8_matmul_example()

    config = copy.deepcopy(matmul_config(mm))
    config.rhs.stats_config.share_stats_axes = [1]
    with self.assertRaisesRegex(aqt_config.ConfigError,
                                "expected rhs matmul contraction axis"):
      self.with_config(mm, config)

    config = copy.deepcopy(matmul_config(mm))
    config.lhs.stats_config.share_stats_axes = [0]
    with self.assertRaisesRegex(aqt_config.ConfigError,
                                "expected lhs matmul contraction axis"):
      self.with_config(mm, config)

  def test_validates_rank2(self):
    mm, lhs, rhs = self.exact_int8_matmul_example()

    mm.rhs_quantizer.data_shape.append(1)
    with self.assertRaisesRegex(aqt_config.ConfigError, "rhs data shape"):
      mm.apply(lhs, rhs)
    mm.rhs_quantizer.data_shape = mm.rhs_quantizer.data_shape[:-1]

    mm.lhs_quantizer.data_shape += (1,)
    with self.assertRaisesRegex(aqt_config.ConfigError, "lhs data shape"):
      mm.apply(lhs, rhs)

  @parameterized.named_parameters(
      aqt_test_shared_base.generate_unaligned_schedule_intervals())
  def test_unaligned_schedule_intervals(self, lhs_intervals, rhs_intervals):
    bits = 8
    lhs_intervals = [(start, stop, bits) for start, stop in lhs_intervals]
    rhs_intervals = [(start, stop, bits) for start, stop in rhs_intervals]
    lhs_config = config_from_schedule(lhs_intervals)
    rhs_config = config_from_schedule(rhs_intervals)
    config = aqt_config.AqtMatmulConfig(lhs_config, rhs_config)

    with self.assertRaisesRegex(aqt_config.ConfigError,
                                "intervals do not match|config len"):
      aqt_matmul.Matmul(config, (1, 1), (1, 1))

  def test_vars_dont_kill_grads(self):
    mm, lhs, rhs = self.exact_int8_matmul_example()

    unsaved_matmul = mm.apply(lhs, rhs)

    with tf.variable_scope("with_var"):
      mm_q, _, _ = self.exact_int8_matmul_example(True, True)

    saved_matmul = mm_q.apply(lhs, rhs)

    saved_grads = tf.gradients([saved_matmul], [lhs, rhs])
    unsaved_grads = tf.gradients([unsaved_matmul], [lhs, rhs])

    with self.cached_session():
      tf.global_variables_initializer().run()

      for matmul in [mm, mm_q]:
        update_event_count(matmul, 0)

      zipped_grads = zip(saved_grads, unsaved_grads)
      for actual_grad, expected_grad in zipped_grads:
        actual = actual_grad.eval()
        expected = expected_grad.eval()

        self.assertAllEqual(actual, expected)

  @parameterized.parameters([
      dict(lhs_use_quantized_variable=True),
      dict(lhs_use_quantized_variable=False)
  ])
  def test_vars_over_inputs_at_inference(self, lhs_use_quantized_variable):
    rhs_use_quantized_variable = not lhs_use_quantized_variable
    mm, tf_lhs, tf_rhs = self.exact_int8_matmul_example(
        lhs_use_quantized_variable, rhs_use_quantized_variable)
    with tf.variable_scope("no_quantize"):
      mm_train, _, _ = self.exact_int8_matmul_example(
          lhs_use_quantized_variable=False, rhs_use_quantized_variable=False)

    with self.cached_session():
      tf.global_variables_initializer().run()
      update_event_count(mm, 0)
      actual = mm.apply(tf_lhs, tf_rhs, train=False).eval()
      expected = np.zeros_like(actual)
      # Rely on zero initialization for variables as opposed to non-zero inputs.
      self.assertAllEqual(actual, expected)

      # But if train then use input instead.
      actual = mm.apply(tf_lhs, tf_rhs, train=True).eval()

      update_event_count(mm_train, 0)
      expected = mm_train.apply(tf_lhs, tf_rhs)
      self.assertAllEqual(actual, expected)

  def test_float_config_not_save_quantized_var(self):
    mm, lhs, rhs = self.exact_int8_matmul_example(
        lhs_use_quantized_variable=True, rhs_use_quantized_variable=True)

    fc = aqt_config.FloatConfig()
    mm.lhs_quantizer.config.tensor_configs[0].quant_config = fc

    with self.cached_session():
      tf.global_variables_initializer().run()
      event_count = tf.constant(0, tf.int64)
      mm.update_lhs(lhs, None, event_count).run()
      mm.update_rhs(rhs, None, event_count).run()

      actual = mm.apply(lhs, rhs, train=False)
      expected = tf.zeros_like(actual)
      # Although lhs config sets use_quantized_variable to True, lhs has
      # a float config, and thus it uses zero-initialized quantized var.
      self.assertAllEqual(actual, expected)

  def test_exact_grads(self):
    mm, lhs, rhs = self.exact_int8_matmul_example()

    aqt_mm = mm.apply(lhs, rhs)
    aqt_mm_grads = tf.gradients([tf.reduce_sum(aqt_mm**2)], [lhs, rhs])

    mm_exact = tf.matmul(lhs, rhs)
    mm_grads = tf.gradients([tf.reduce_sum(mm_exact**2)], [lhs, rhs])

    with self.cached_session():
      tf.global_variables_initializer().run()
      update_event_count(mm, 0)

      for aqt_grad, tf_grad in zip(aqt_mm_grads, mm_grads):
        actual = aqt_grad.eval()
        expected = tf_grad.eval()

        self.assertAllEqual(actual, expected)

  def test_grad_linearity(self):
    """Validates gradients are correct on basic example."""
    float_config_tc = aqt_config.AqtTensorConfig(
        freeze_scale_at_begin=True,
        quant_config=aqt_config.FloatConfig(),
        calibration_config=calibration_config(1))
    float_config = aqt_config.AqtScheduleConfig(test_stats_config(),
                                                [float_config_tc])
    scale = 10.0
    int_config = _schedule_config(8, scale, (0, 1))

    lhs_config, rhs_config = int_config, float_config
    contract_dim = 10
    lhs_shape = (1, contract_dim)
    rhs_shape = (contract_dim, 1)
    target_shape = lhs_shape[:1] + rhs_shape[1:]

    lhs_ph = tf.placeholder(tf.float32, shape=lhs_shape)
    rhs_ph = tf.placeholder(tf.float32, shape=rhs_shape)
    target_ph = tf.placeholder(tf.float32, shape=target_shape)

    config = aqt_config.AqtMatmulConfig(lhs_config, rhs_config)
    mm = aqt_matmul.Matmul(config, lhs_shape, rhs_shape)

    with self.cached_session() as sess, sess.as_default():
      tf.global_variables_initializer().run()

      event_count = tf.constant(0, tf.int64)
      updates = [
          mm.update_lhs(tf.ones(lhs_shape), None, event_count),
          mm.update_rhs(tf.ones(rhs_shape), None, event_count)
      ]
      with tf.control_dependencies(updates):
        aqt_mm = mm.apply(lhs_ph, rhs_ph)

      aqt_diff = aqt_mm - target_ph
      aqt_loss = tf.reduce_sum(aqt_diff**2) / 2
      aqt_mm_grad = tf.gradients([aqt_loss], [rhs_ph])[0]

      rng = np.random.default_rng(1234)
      for i in range(10):
        lhs = rng.standard_normal(lhs_shape).astype(np.float32)
        rhs = rng.standard_normal(rhs_shape).astype(np.float32)
        target = rng.standard_normal(target_shape).astype(np.float32)

        feed_dict = {lhs_ph: lhs, rhs_ph: rhs, target_ph: target}

        aqtd, aqt_grad = sess.run([aqt_diff, aqt_mm_grad], feed_dict=feed_dict)

        # Notice aqt gradient at position i is quantized(lhs)[i] * aqtd
        # assuming linearity of gradients.
        grad_factor = aqtd.ravel()
        float_grad = lhs.ravel() * grad_factor
        true_grad = aqt_grad.ravel()
        diff = np.abs(float_grad - true_grad)
        bucket_width = scale * 2 / 255
        for j, err in enumerate(diff):
          self.assertLessEqual(
              err,
              bucket_width * abs(grad_factor),
              msg=f"trial {i} position {j}")

  def test_diagnostics(self):
    mm, lhs, rhs = self.exact_int8_matmul_example()

    with self.cached_session():
      tf.global_variables_initializer().run()
      update_event_count(mm, 0)

      d = mm.diagnostics(lhs, rhs)
      quantizers = {"lhs": mm.lhs_quantizer, "rhs": mm.rhs_quantizer}
      for qname, quantizer in quantizers.items():

        for name, expected in quantizer.calibration_variables().items():
          actual = d[f"aqt/{qname}/{name}"]
          self.assertAllEqual(actual, expected)

        actual = d[f"aqt/{qname}/clipped_proportion"]
        expected = 0.0
        self.assertAllEqual(actual, expected)

        actual = d[f"aqt/{qname}/clip"]
        expected = quantizer.clip_range()
        self.assertAllEqual(actual, expected)

      out_of_range_lhs, out_of_range_rhs = (
          tf.ones_like(x) * 512.0 for x in (lhs, rhs))
      d = mm.diagnostics(out_of_range_lhs, out_of_range_rhs)
      for arg in ["lhs", "rhs"]:
        actual = d[f"aqt/{arg}/clipped_proportion"]
        expected = 1.0
        self.assertAllEqual(actual, expected)


if __name__ == "__main__":
  absltest.main()
