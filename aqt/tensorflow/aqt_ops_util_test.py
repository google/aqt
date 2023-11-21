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

"""Tests for utility function."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from aqt.tensorflow import aqt_ops_util
from aqt.tensorflow import aqt_tensor
from aqt.test import aqt_matmul_test_base
import tensorflow.compat.v1 as tf


class IntNarrowedMatMulTest(tf.test.TestCase, parameterized.TestCase):

  def test_chooses_right_matmul_for_schedule(self):
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

    # any config that does not raise an error
    lhs_config = aqt_matmul_test_base.config_from_schedule(lhs_schedule)
    rhs_config = aqt_matmul_test_base.config_from_schedule(rhs_schedule)

    shape = [1, 1]  # Any shape will do, we're mocking.
    lhs_quant = aqt_tensor.TensorQuantizer(shape, lhs_config, name="lhs")
    rhs_quant = aqt_tensor.TensorQuantizer(shape, rhs_config, name="rhs")

    default_fn = mock.Mock(return_value=tf.constant("default"))
    int8_fn = mock.Mock(return_value=tf.constant("int8"))

    event_ph = tf.placeholder(tf.int64)
    lhs_quant._last_update = event_ph
    rhs_quant._last_update = event_ph
    train = True
    tf_actual = aqt_ops_util._dense_op_case(
        lhs_quant, rhs_quant, default_fn, int8_fn, train
    )

    with self.cached_session():
      tf.global_variables_initializer().run()
      for i, expected in enumerate(expected_results):
        actual = tf_actual.eval(feed_dict={event_ph: i})
        self.assertEqual(
            actual.decode("utf-8"), expected, msg=f"event_count {i}"
        )

  def test_quant_all_the_time(self):
    # Create a list of settings (left_bits, right_bits, expected_matmul)
    # and generate a schedule based on those settings.
    settings = [(8, 8, "int8"), (8, 8, "int8")]

    lhs_schedule = []
    rhs_schedule = []
    expected_results = []
    for i, (l, r, expected) in enumerate(settings):
      lhs_schedule.append((i, i + 1, l))
      rhs_schedule.append((i, i + 1, r))
      expected_results.append(expected)

    # any config that does not raise an error
    lhs_config = aqt_matmul_test_base.config_from_schedule(lhs_schedule)
    rhs_config = aqt_matmul_test_base.config_from_schedule(rhs_schedule)

    shape = [1, 1]  # Any shape will do, we're mocking.
    lhs_quant = aqt_tensor.TensorQuantizer(shape, lhs_config, name="lhs")
    rhs_quant = aqt_tensor.TensorQuantizer(shape, rhs_config, name="rhs")

    default_fn = mock.Mock(return_value=tf.constant("default"))
    int8_fn = mock.Mock(return_value=tf.constant("int8"))

    event_ph = tf.placeholder(tf.int64)
    lhs_quant._last_update = event_ph
    rhs_quant._last_update = event_ph
    train = True
    tf_actual = aqt_ops_util._dense_op_case(
        lhs_quant, rhs_quant, default_fn, int8_fn, train
    )

    with self.cached_session():
      tf.global_variables_initializer().run()
      for i, expected in enumerate(expected_results):
        actual = tf_actual.eval(feed_dict={event_ph: i})
        self.assertEqual(
            actual.decode("utf-8"), expected, msg=f"event_count {i}"
        )


if __name__ == "__main__":
  absltest.main()
