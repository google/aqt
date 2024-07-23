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


class AqtNumerics(abc.ABC):
  """Numerics for int8, int4, binary, etc."""

  # TODO(lew): Currently this is a part of API, only because it is used to set
  # it in test. Remove and leave only get_dtype(

  @abc.abstractmethod
  def get_dtype(self):
    pass

  @abc.abstractmethod
  def get_scaled_bound(self):
    """Returns the width that the scale corresponds to in the quantizion range.

    For symmetric scaling (relative to a fixed zero point) it could be biggest
    value that can be represented by numerical format exactly. E.g. in case of
    int8, 127 . Or it could be edge of the last bucket (in case of int8, 127.5).

    For asymmetric scaling, it corresponds to the width of the entire
    quantization range. E.g. in case of int8, 255.
    """
    pass

  @abc.abstractmethod
  def vjp_fwd(self, x, context):
    pass

  @abc.abstractmethod
  def vjp_bwd(self, res, grad):
    pass
