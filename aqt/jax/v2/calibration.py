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
"""Quantization calibration methods."""

from typing import Optional
import flax.struct
import jax.numpy as jnp


@flax.struct.dataclass
class AbsMaxCalibration:
  """Simple max(abs(x)) calibration."""

  bound: Optional[jnp.ndarray] = None

  def get_bound(self, x, shared_axes) -> jnp.ndarray:
    """Calibration."""
    msg = 'Perhaps you are using fake_quant and forgot to set them.'
    assert shared_axes is not None, msg

    if self.bound is None:
      # NOTE: If you want to clip, modify _make_int_quant.vjp_bwd.
      abs_max = jnp.max(jnp.abs(x), axis=shared_axes, keepdims=True)
    else:
      assert self.bound > 0, 'Static quantization bound should be positive.'
      abs_max = jnp.asarray(self.bound).reshape((1,) * len(x.shape))
    abs_max = jnp.where(abs_max == 0.0, jnp.ones_like(abs_max), abs_max)
    return abs_max
