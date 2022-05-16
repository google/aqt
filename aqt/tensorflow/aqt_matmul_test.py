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
from aqt.tensorflow import aqt_ops
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


def update_event_count(quantizer, event_count: int):
  """Update the quantizer's event count without changing stats."""
  sample = tf.zeros(quantizer.data_shape)
  weights = tf.zeros([1] * len(quantizer.data_shape))
  event_count = tf.constant(event_count, tf.int64)
  quantizer.update(sample, weights, event_count).run()


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

    lhs_quantizer = aqt_tensor.TensorQuantizer(
        lhs.shape, lhs_config, name="lhs")
    rhs_quantizer = aqt_tensor.TensorQuantizer(
        rhs.shape, rhs_config, name="rhs")

    return lhs_quantizer, lhs, rhs_quantizer, rhs

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
    lhs_tq = aqt_tensor.TensorQuantizer(lhs.shape, no_quant_config, name="lhs")
    lhs_float_tq = aqt_tensor.TensorQuantizer(
        lhs.shape, float_config, name="lhs_float")
    lhs = tf.constant(lhs)

    rhs = np.random.uniform(-1.0, 1.0, size=(3, 4)).astype(np.float32)
    rhs_tq = aqt_tensor.TensorQuantizer(rhs.shape, no_quant_config, name="rhs")
    rhs_float_tq = aqt_tensor.TensorQuantizer(
        rhs.shape, float_config, name="rhs_float")
    rhs = tf.constant(rhs)

    no_quant_ret = aqt_ops.aqt_matmul(lhs_tq, lhs, rhs_tq, rhs)
    float_config_ret = aqt_ops.aqt_matmul(lhs_float_tq, lhs, rhs_float_tq, rhs)
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

    return lhs_config, lhs, qlhs, rhs_config, rhs, qrhs

  def test_basic_matmul(self):
    lhs_config, lhs, qlhs, rhs_config, rhs, qrhs = self.basic_quant_example()

    lhs_tq = aqt_tensor.TensorQuantizer(lhs.shape, lhs_config, name="lhs")
    rhs_tq = aqt_tensor.TensorQuantizer(rhs.shape, rhs_config, name="rhs")

    event_count = tf.constant(0, tf.int64)
    updates = [
        lhs_tq.update(lhs, None, event_count),
        rhs_tq.update(rhs, None, event_count)
    ]
    with tf.control_dependencies(updates):
      actual = aqt_ops.aqt_matmul(lhs_tq, lhs, rhs_tq, rhs)
    expected = tf.matmul(qlhs, qrhs)

    with self.cached_session() as sess, sess.as_default():
      tf.global_variables_initializer().run()
      self.assertAllEqual(expected, actual)

  @parameterized.parameters([dict(lhs_float=True), dict(lhs_float=False)])
  def test_float_config_basic_matmul(self, lhs_float):
    lhs_config, lhs, qlhs, rhs_config, rhs, qrhs = self.basic_quant_example()

    if lhs_float:
      lhs_config.tensor_configs[0].quant_config = aqt_config.FloatConfig()
    else:
      rhs_config.tensor_configs[0].quant_config = aqt_config.FloatConfig()

    lhs_tq = aqt_tensor.TensorQuantizer(lhs.shape, lhs_config, name="lhs")
    rhs_tq = aqt_tensor.TensorQuantizer(rhs.shape, rhs_config, name="rhs")

    event_count = tf.constant(0, tf.int64)
    updates = [
        lhs_tq.update(lhs, None, event_count),
        rhs_tq.update(rhs, None, event_count)
    ]
    with tf.control_dependencies(updates):
      actual = aqt_ops.aqt_matmul(lhs_tq, lhs, rhs_tq, rhs)

    if lhs_float:
      qlhs = lhs  # lhs is not quantized
    else:
      qrhs = rhs  # rhs is not quantized
    expected = tf.matmul(qlhs, qrhs)

    with self.cached_session() as sess, sess.as_default():
      tf.global_variables_initializer().run()
      self.assertAllEqual(expected, actual)

  def with_config(self, quantizer, config):
    """Returns new quantizer with the new config but otherwise the same."""
    with tf.variable_scope(None, default_name="uniqued"):
      return aqt_tensor.TensorQuantizer(quantizer.data_shape, config)

  def test_validates_contraction(self):
    lhs_quantizer, lhs, rhs_quantizer, rhs = self.exact_int8_matmul_example()

    config = copy.deepcopy(rhs_quantizer.config)
    config.stats_config.share_stats_axes = [1]
    bad_rhs_quantizer = self.with_config(rhs_quantizer, config)
    with self.assertRaisesRegex(aqt_config.ConfigError,
                                "expected rhs matmul contraction axis"):
      aqt_ops.aqt_matmul(lhs_quantizer, lhs, bad_rhs_quantizer, rhs)

    config = copy.deepcopy(lhs_quantizer.config)
    config.stats_config.share_stats_axes = [0]
    bad_lhs_quantizer = self.with_config(rhs_quantizer, config)
    with self.assertRaisesRegex(aqt_config.ConfigError,
                                "expected lhs matmul contraction axis"):
      aqt_ops.aqt_matmul(bad_lhs_quantizer, lhs, rhs_quantizer, rhs)

  def test_validates_rank2(self):
    lhs_quantizer, lhs, rhs_quantizer, rhs = self.exact_int8_matmul_example()

    rhs_quantizer.data_shape.append(1)
    with self.assertRaisesRegex(aqt_config.ConfigError, "rhs data shape"):
      aqt_ops.aqt_matmul(lhs_quantizer, lhs, rhs_quantizer, rhs)
    rhs_quantizer.data_shape = rhs_quantizer.data_shape[:-1]

    lhs_quantizer.data_shape += (1,)
    with self.assertRaisesRegex(aqt_config.ConfigError, "lhs data shape"):
      aqt_ops.aqt_matmul(lhs_quantizer, lhs, lhs_quantizer, lhs)

  @parameterized.named_parameters(
      aqt_test_shared_base.generate_unaligned_schedule_intervals())
  def test_unaligned_schedule_intervals(self, lhs_intervals, rhs_intervals):
    bits = 8
    lhs_intervals = [(start, stop, bits) for start, stop in lhs_intervals]
    rhs_intervals = [(start, stop, bits) for start, stop in rhs_intervals]
    lhs_config = config_from_schedule(lhs_intervals)
    rhs_config = config_from_schedule(rhs_intervals)

    lhs = rhs = np.ones((1, 1)).astype(np.float32)
    lhs_quantizer = aqt_tensor.TensorQuantizer(
        lhs.shape, lhs_config, name="lhs")
    rhs_quantizer = aqt_tensor.TensorQuantizer(
        rhs.shape, rhs_config, name="rhs")
    with self.assertRaisesRegex(aqt_config.ConfigError,
                                "intervals do not match|config len"):
      lhs = tf.constant(lhs)
      rhs = tf.constant(rhs)
      aqt_ops.aqt_matmul(lhs_quantizer, lhs, rhs_quantizer, rhs)

  def test_vars_dont_kill_grads(self):
    lhs_q_novar, lhs, rhs_q_novar, rhs = self.exact_int8_matmul_example()

    unsaved_matmul = aqt_ops.aqt_matmul(lhs_q_novar, lhs, rhs_q_novar, rhs)

    with tf.variable_scope("with_var"):
      lhs_q, _, rhs_q, _ = self.exact_int8_matmul_example(True, True)

    saved_matmul = aqt_ops.aqt_matmul(lhs_q, lhs, rhs_q, rhs)

    saved_grads = tf.gradients([saved_matmul], [lhs, rhs])
    unsaved_grads = tf.gradients([unsaved_matmul], [lhs, rhs])

    with self.cached_session():
      tf.global_variables_initializer().run()

      for q in [lhs_q_novar, lhs_q, rhs_q_novar, rhs_q]:
        update_event_count(q, 0)

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
    lhs_quantizer, tf_lhs, rhs_quantizer, tf_rhs = self.exact_int8_matmul_example(
        lhs_use_quantized_variable, rhs_use_quantized_variable)
    with tf.variable_scope("no_quantize"):
      lhs_quantizer_train, _, rhs_quantizer_train, _ = self.exact_int8_matmul_example(
          lhs_use_quantized_variable=False, rhs_use_quantized_variable=False)

    mm = aqt_ops.aqt_matmul(lhs_quantizer, tf_lhs, rhs_quantizer, tf_rhs,
                            train=False)

    with self.cached_session():
      tf.global_variables_initializer().run()
      update_event_count(lhs_quantizer, 0)
      update_event_count(rhs_quantizer, 0)
      actual = mm.eval()
      expected = np.zeros_like(actual)
      # Rely on zero initialization for variables as opposed to non-zero inputs.
      self.assertAllEqual(actual, expected)

      # But if train then use input instead.
      actual = aqt_ops.aqt_matmul(lhs_quantizer, tf_lhs, rhs_quantizer, tf_rhs,
                                  train=True)
      update_event_count(lhs_quantizer_train, 0)
      update_event_count(rhs_quantizer_train, 0)
      expected = aqt_ops.aqt_matmul(lhs_quantizer_train, tf_lhs,
                                    rhs_quantizer_train, tf_rhs)
      self.assertAllEqual(actual, expected)

  def test_float_config_not_save_quantized_var(self):
    lhs_quantizer, lhs, rhs_quantizer, rhs = self.exact_int8_matmul_example(
        lhs_use_quantized_variable=True, rhs_use_quantized_variable=True)

    fc = aqt_config.FloatConfig()
    lhs_quantizer.config.tensor_configs[0].quant_config = fc

    with self.cached_session():
      tf.global_variables_initializer().run()
      event_count = tf.constant(0, tf.int64)
      lhs_quantizer.update(lhs, None, event_count).run()
      rhs_quantizer.update(rhs, None, event_count).run()

      actual = aqt_ops.aqt_matmul(
          lhs_quantizer, lhs, rhs_quantizer, rhs, train=False)
      expected = tf.zeros_like(actual)
      # Althoguh lhs config sets use_quantized_variable to True, lhs has
      # a float config, and thus it uses zero-initialized quantized var.
      self.assertAllEqual(actual, expected)

  def test_exact_grads(self):
    lhs_quantizer, lhs, rhs_quantizer, rhs = self.exact_int8_matmul_example()

    aqt_mm = aqt_ops.aqt_matmul(lhs_quantizer, lhs, rhs_quantizer, rhs)
    aqt_mm_grads = tf.gradients([tf.reduce_sum(aqt_mm**2)], [lhs, rhs])

    mm = tf.matmul(lhs, rhs)
    mm_grads = tf.gradients([tf.reduce_sum(mm**2)], [lhs, rhs])

    with self.cached_session():
      tf.global_variables_initializer().run()
      update_event_count(lhs_quantizer, 0)
      update_event_count(rhs_quantizer, 0)

      for aqt_grad, tf_grad in zip(aqt_mm_grads, mm_grads):
        actual = aqt_grad.eval()
        expected = tf_grad.eval()

        self.assertAllEqual(actual, expected)


if __name__ == "__main__":
  absltest.main()
