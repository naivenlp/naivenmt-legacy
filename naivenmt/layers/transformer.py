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
                        scope="positional_encoding"):
  """Positional encoding as described in https://arxiv.org/abs/1706.03762.

  Args:
    inputs: A 2-d tensor with shape [N, T]
    num_units: Output dimension, or the model's dimension
    scope: Variable scope
  """
  batch_size, time_steps = inputs.get_shape().as_list()
  with tf.variable_scope(scope):
    position_index = tf.tile(
      tf.expand_dims(tf.range(time_steps), 0), [batch_size, 1])

    position_encoding = np.array([
      [pos / np.power(10000, 2. * i / num_units) for i in range(num_units)]
      for pos in range(time_steps)])
    position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])  # dim 2i
    position_encoding[:, 1::2] = np.cos(
      position_encoding[:, 1::2])  # dim 2i+1

    # Convert to a tensor
    lookup_table = tf.convert_to_tensor(position_encoding)

    outputs = tf.nn.embedding_lookup(lookup_table, position_index)

    return outputs


def norm(inputs):
  return tf.layers.batch_normalization(inputs=inputs)


def add_and_norm(x, sub_x, dropout=0.2):
  with tf.variable_scope("add_and_norm"):
    sub_x = tf.nn.dropout(sub_x, 1 - dropout)
    return tf.layers.batch_normalization(inputs=x + sub_x)


def scaled_dot_product_attention(q, k, v, scale=None, mask=None, dropout=0.2):
  """Scaled dot-product attention.

  Args:
    q: Query tensor, with shape [B, L_q, D]
    k: Key tensor, with shape [N, L_k, D]
    v: Value tensor, with shape [N, L_v, D]
    scale: A scalar, scale factor
    mask: Attention mask, with shape [B, L, L]
    dropout: A scalar, dropout rate
  """
  dot = tf.matmul(q, k, transpose_b=True)
  if scale:
    dot = dot * scale
  if mask:
    padding = tf.ones_like(dot) * dot.dtype.min
    dot = tf.where(tf.equal(mask, 0), padding, dot)
  attention = tf.nn.softmax(dot)
  attention = tf.nn.dropout(attention, dropout)
  output = tf.matmul(attention, v)
  return output, attention


def multihead_attention(queries,
                        keys,
                        values,
                        num_heads=8,
                        dropout=0.2,
                        mask=None,
                        scope="multihead_attention"):
  """Multi-head attention mechanism.

  Args:
    queries: Query tensor, with shape [B, L_q, D]
    keys: Key tensor, with shape [B, L_k, D]
    values: Value tensor, with shape [B, L_v, D]
    num_heads: A scalar, number of heads to split
    dropout: A scalar, dropout rate.
    mask: Making tensor, with shape [B, L_q, L_k]
    scope: A string, variable scope name.
  """
  with tf.variable_scope(scope) as scope:
    model_dim = queries.get_shape()[-1]

    q = tf.layers.dense(
      queries, model_dim, activation=tf.nn.relu)  # (B, L_q, D]
    k = tf.layers.dense(
      keys, model_dim, activation=tf.nn.relu)
    v = tf.layers.dense(
      values, model_dim, activation=tf.nn.relu)

    # split and concat
    q = tf.concat(tf.split(q, num_heads, axis=2), 0)  # [h*B, L_q, D/h]
    k = tf.concat(tf.split(k, num_heads, axis=2), 0)
    v = tf.concat(tf.split(v, num_heads, axis=2), 0)

    scale = (model_dim // num_heads) ** -0.5
    output, attention = scaled_dot_product_attention(
      q, k, v, scale, mask, dropout)

    output = tf.concat(tf.split(output, num_heads, axis=0), 2)
    output = tf.layers.dense(output, model_dim)
    output = tf.nn.dropout(output, dropout)

    # residual
    output += queries
    # layer norm
    output = norm(output)

    return output, attention


def positional_wise_feed_forward_network(inputs,
                                         model_dim=512,
                                         ffn_dim=2048,
                                         dropout=0.2,
                                         scope="ffn"):
  with tf.variable_scope(scope) as scope:
    params = {"inputs": inputs, "filters": model_dim, "kernel_size": 1,
              "activation": tf.nn.relu, "use_bias": True}
    outputs = tf.layers.conv1d(**params)

    # Readout layer
    params = {"inputs": outputs, "filters": ffn_dim, "kernel_size": 1,
              "activation": None, "use_bias": True}
    outputs = tf.layers.conv1d(**params)

    outputs = tf.layers.dropout(outputs, dropout)

    # residual and layer norm
    outputs += inputs
    outputs = norm(outputs)

    return outputs
