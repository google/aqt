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
"""Base abstract class for all numerics."""
import abc

from aqt.jax.v2 import utils


@utils.flax_slots_kw_only_dataclass
class AqtNumerics(abc.ABC):
  """Abstract class for various quantization numerics."""

  @abc.abstractmethod
  def get_dtype(self):
    pass

  @abc.abstractmethod
  def get_quant_bound(self):
    """The width that the bound corresponds to in the quantization range.

    For symmetric scaling (relative to a fixed zero point), the bound represents
    the largest absolute value that should be representable in the input tensor,
    and the quant_bound represents where that corresponds to in the
    quantization range.

    It could be biggest value that can be represented by numerical format
    exactly (e.g. in case of int8, 127), or it could be edge of the last bucket
    (e.g. in case of int8, 127.5. In this case the largest representable
    absolute value will be slightly smaller than the bound).

    For asymmetric scaling, the bound corresponds to the width of the entire
    input tensor, and the quant_bound corresponds to the width of the entire
    quantization range (e.g. in case of int8, 255).
    """
    pass

  @abc.abstractmethod
  def vjp_fwd(self, x, context):
    pass

  @abc.abstractmethod
  def vjp_bwd(self, res, grad):
    pass
