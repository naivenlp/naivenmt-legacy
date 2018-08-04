# Copyright 2018 luozhouyang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import abc


class ModelTensorsHook(abc.ABC):
  """Listen tensors state when it has been created."""

  @abc.abstractmethod
  def on_global_steps_created(self, global_steps):
    """Listen global_steps.

    Args:
      global_steps: global steps.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def on_learning_rate_created(self, learning_rate):
    """Listen learning_rate.

    Args:
      learning_rate: learning rate.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def on_gradients_created(self, gradients):
    """Listen gradients before clipping.

    Args:
      gradients: gradients.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def on_gradients_clipped(self, clipped_grads, grad_norm):
    """Listen states of clipped gradients.

    Args:
      clipped_grads: clipped gradients.
      grad_norm: gradients norm.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def on_optimizer_created(self, optimizer):
    """Listen optimizer.

    Args:
      optimizer: optimizer
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def on_loss_created(self, loss):
    """Listen loss.

    Args:
      loss
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def end(self):
    """Called after all above tensors being created."""
    raise NotImplementedError()
