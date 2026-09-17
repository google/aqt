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

"""Class of different sparsity modes."""

import abc
import dataclasses
from jax import numpy as jnp


@dataclasses.dataclass
class BaseSparsityScheduleMode(abc.ABC):
  """Base class of Sparsity schedule mode defining update and apply condition interfaces."""

  @abc.abstractmethod
  def update_cond(self, step: int, mask_update_times: int) -> jnp.ndarray:
    """Sparse mask update schedule.

    Make mask update when following conditions are all satisfied.

    Args:
      step: current step.
      mask_update_times: number of times sparse masks have been updated.

    Returns:
      A boolean condition result indicating whether to update the mask at the
      current step.
    """
    pass

  @abc.abstractmethod
  def get_num_shots(self) -> int:
    pass


@dataclasses.dataclass
class TrainingMode(BaseSparsityScheduleMode):
  """Collection of hyper-parameters for sparsity awareness training mode.

  Attributes:
    target_step: target step to start sparsity pruning.
  """

  target_step: int = 0

  def update_cond(self, step: int, mask_update_times: int) -> jnp.ndarray:
    """Sparse mask update schedule.

    Make mask update when current step is the target step to do pruning.

    Args:
      step: current step.
      mask_update_times: number of times sparse masks have been updated.

    Returns:
      A boolean condition result indicating whether to update the mask at the
      current step.
    """
    return jnp.greater_equal(step, self.target_step)

  def get_num_shots(self) -> int:
    return -1


@dataclasses.dataclass
class OneShotMode(BaseSparsityScheduleMode):
  """Collection of hyper-parameters for sparsity one shot training mode.

  Attributes:
    target_step: target step to start sparsity pruning.
  """

  target_step: int = 0

  def update_cond(self, step: int, mask_update_times: int) -> jnp.ndarray:
    """Sparse mask update schedule.

    Make mask update when 1) mask_update_times < 1 and 2) current step is
    the target step to do pruning.

    Args:
      step: current step.
      mask_update_times: number of times sparse masks have been updated.

    Returns:
      A boolean condition result indicating whether to update the mask at the
      current step.
    """
    should_do_pruning = jnp.less(mask_update_times, 1)
    should_pruning_step = jnp.equal(step, self.target_step)
    return jnp.logical_and(should_pruning_step, should_do_pruning)

  def get_num_shots(self) -> int:
    return 1


@dataclasses.dataclass
class MaterializeMode(BaseSparsityScheduleMode):
  """Sparsity awareness materialize mode."""

  def update_cond(self, step: int, mask_update_times: int) -> jnp.ndarray:
    """Sparse mask update schedule.

    Never update mask for Materialize mode.

    Args:
      step: current step.
      mask_update_times: number of times sparse masks have been updated.

    Returns:
      A boolean condition result indicating whether to update the mask at the
      current step.
    """

    return jnp.bool_(False)

  def get_num_shots(self) -> int:
    return -1


@dataclasses.dataclass
class InferenceMode(BaseSparsityScheduleMode):
  """Sparsity awareness inference mode."""

  def update_cond(self, step: int, mask_update_times: int) -> jnp.ndarray:
    """Sparse mask update schedule.

    Never update mask for Inference mode.

    Args:
      step: current step.
      mask_update_times: number of times sparse masks have been updated.

    Returns:
      A boolean condition result indicating whether to update the mask at the
      current step.
    """
    return jnp.bool_(False)

  def get_num_shots(self) -> int:
    return 0


@dataclasses.dataclass
class FewShotMode(BaseSparsityScheduleMode):
  """Collection of hyper-parameters for sparsity few shot training mode.

  Attributes:
    num_shots: Number of shots during pruning. This needs to be set in FEWSHOT
      mode.
    mask_update_interval: The step invertal between two mask updates. This is
      only valid under FEWSHOT mode.
    target_step: initial target step to start sparsity pruning.
    next_target_step: next target step to do sparsity pruning.
  """

  num_shots: int = 0
  mask_update_interval: int = 1
  target_step: int = 0
  next_target_step: int = 0

  def update_cond(self, step: int, mask_update_times: int) -> jnp.ndarray:
    """Sparse mask update schedule.

    Make mask update when 1) mask_update_times < num_shots and 2) current step
    is
    the target step to do pruning.

    Args:
      step: current step.
      mask_update_times: number of times sparse masks have been updated.

    Returns:
      A boolean condition result indicating whether to update the mask at the
      current step.
    """
    should_do_pruning = jnp.less(mask_update_times, self.num_shots)
    should_pruning_step = jnp.equal(step, self.next_target_step)
    return jnp.logical_and(should_pruning_step, should_do_pruning)

  def increment_target_step(self, mask_update_times: int) -> None:
    self.next_target_step = (
        self.target_step + mask_update_times * self.mask_update_interval
    )

  def get_num_shots(self) -> int:
    return self.num_shots

  def __post_init__(self):
    assert self.num_shots > 1, '`num_shots should be set for FEWSHOT sparse.`'
    self.next_target_step = self.target_step
