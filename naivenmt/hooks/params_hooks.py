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


class CountParamsHook(tf.train.SessionRunHook):
  """Logs the number of trainable parameters."""

  def begin(self):
    total = 0
    for variable in tf.trainable_variables():
      shape = variable.get_shape()
      count = 1
      for dim in shape:
        count *= dim.value
      total += count
    tf.logging.info("Number of trainable parameters: %d", total)
