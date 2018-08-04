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

import tensorflow as tf

from naivenmt.hooks.model_tensors_hooks import ModelTensorsHook


class TensorsCollectionHook(ModelTensorsHook):
  """Add tensors to graph's collection."""

  def __init__(self):
    self.tensors = {}

  def on_global_steps_created(self, global_steps):
    # do nothing, cause global steps has been added to collections
    pass

  def on_learning_rate_created(self, learning_rate):
    self.tensors["learning_rate"] = learning_rate

  def on_gradients_created(self, gradients):
    self.tensors["gradients"] = gradients

  def on_gradients_clipped(self, clipped_grads, grad_norm):
    self.tensors["clipped_grads"] = clipped_grads
    self.tensors["grad_norm"] = grad_norm

  def on_optimizer_created(self, optimizer):
    # we do not collect optimizer
    pass

  def on_loss_created(self, loss):
    self.tensors["loss"] = loss

  def end(self):
    for k, v in self.tensors.items():
      tf.add_to_collection(k, v)
    tf.logging.info("Tensors are added to collections.")
