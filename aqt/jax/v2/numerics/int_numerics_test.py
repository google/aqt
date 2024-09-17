# Copyright 2024 Google LLC
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

import operator

from absl.testing import parameterized
from aqt.jax.v2 import aqt_quantizer
from aqt.jax.v2 import calibration
from aqt.jax.v2 import utils
from aqt.jax.v2.numerics import int_numerics
import jax.numpy as jnp
import numpy as np

from google3.testing.pybase import googletest


def inclusive_arange(start, stop, step):
  return jnp.arange(start, stop + step, step)


def assert_array_less_or_equal(x, y, err_msg='', verbose=True):
  np.testing.assert_array_compare(
      operator.__le__,
      x,
      y,
      err_msg=err_msg,
      verbose=verbose,
      header='x is not less than or equal to y',
      equal_inf=False,
  )


def assert_array_greater_or_equal(x, y, err_msg='', verbose=True):
  np.testing.assert_array_compare(
      operator.__ge__,
      x,
      y,
      err_msg=err_msg,
      verbose=verbose,
      header='x is not greater than or equal to y',
      equal_inf=False,
  )


class IntSymmetricTest(parameterized.TestCase):

  @parameterized.product(
      bits=[2, 4],
      preserve_zero=[False, True],
      preserve_max_val=[False, True],
  )
  def test_quant_range(
      self,
      bits,
      preserve_zero,
      preserve_max_val,
  ):
    sint_min = -(2 ** (bits - 1))
    sint_max = 2 ** (bits - 1) - 1
    sint_min_restricted = sint_min + 1

    def quantize(x):
      numerics_ = int_numerics.IntSymmetric(
          bits=bits,
          preserve_zero=preserve_zero,
          preserve_max_val=preserve_max_val,
          # The quantized values are only guarenteed to be within the
          # appropriate signed int range if clip=True and round=True.
          clip=True,
          clip_gradient=False,
          round=True,
          noise_fn=None,
          dtype=jnp.int8,
      )
      q = aqt_quantizer.Quantizer(
          numerics=numerics_,
          calib_shared_axes=None,
          scale_stop_grad=True,
          calibration=calibration.AbsMaxCalibration,
          context=utils.Context(key=None, train_step=None),
      )
      q.init_calibration()
      qx, _ = q.quant(x, calibration_axes=-1)
      return qx

    step_size = 0.25
    qx_neg_max_abs = quantize(inclusive_arange(-2, 1, step_size))
    qx_eq_max_abs = quantize(inclusive_arange(-1, 1, step_size))
    qx_pos_max_abs = quantize(inclusive_arange(-1, 2, step_size))

    # Test that the quantized value is not outside of the signed int range.
    # E.g. [-2, 1] for bits=2.
    assert_array_less_or_equal(qx_neg_max_abs.qvalue, sint_max)
    assert_array_less_or_equal(qx_eq_max_abs.qvalue, sint_max)
    assert_array_less_or_equal(qx_pos_max_abs.qvalue, sint_max)
    assert_array_greater_or_equal(qx_neg_max_abs.qvalue, sint_min)
    assert_array_greater_or_equal(qx_eq_max_abs.qvalue, sint_min)
    assert_array_greater_or_equal(qx_pos_max_abs.qvalue, sint_min)

    # Test that the quantized value is in the restricted symmetric range.
    # E.g. -127, 127 for bits=8 and [-1, 1] for bits=2.
    assert_array_greater_or_equal(qx_neg_max_abs.qvalue, sint_min_restricted)
    assert_array_greater_or_equal(qx_eq_max_abs.qvalue, sint_min_restricted)
    assert_array_greater_or_equal(qx_pos_max_abs.qvalue, sint_min_restricted)


if __name__ == '__main__':
  googletest.main()
