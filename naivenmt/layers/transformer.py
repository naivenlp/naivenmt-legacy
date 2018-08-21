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

import numpy as np
import tensorflow as tf


# This fucntion is modified from https://github.com/Kyubyong/transformer/blob/master/modules.py
#  with Apache License V2
def positional_encoding(inputs,
                        num_units,
                        zero_pad=True,
                        scale=True,
                        scope="positional_encoding"):
  """Positional encoding as described in https://arxiv.org/abs/1706.03762.

  Args:
    inputs: A 2-d tensor with shape [N, T]
    num_units: Output dimension
    zero_pad:
  """
  N, T = inputs.get_shape().as_list()  # batch_size, time_steps
  with tf.variable_scope(scope):
    position_index = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

    position_encoding = np.array([
      [pos / np.power(10000, 2. * i / num_units) for i in range(num_units)]
      for pos in range(T)])
    position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])  # dim 2i
    position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])  # dim 2i+1

    # Convert to a tensor
    lookup_table = tf.convert_to_tensor(position_encoding)

    if zero_pad:
      lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                lookup_table[1:, :]), 0)
    outputs = tf.nn.embedding_lookup(lookup_table, position_index)

    if scale:
      outputs = outputs * num_units ** 0.5

    return outputs
