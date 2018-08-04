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


class IteratorHooksCreator(object):
  """Create hooks for train and eval iterator."""

  def create_batching_func(self, batch_size, src_eos_id, tgt_eos_id):
    """Create a batching_func.

    Args:
      batch_size: batch size
      src_eos_id: source eos id
      tgt_eos_id: target eos id

    Returns:
      batching_func: a func with signature (x) and
        return a padded batch of dataset `x`
    """
    raise NotImplementedError()

  def create_key_func(self, src_max_len, num_buckets):
    """Decide a bucket to save tokens.

    Args:
      src_max_len: maximum source sequence length
      num_buckets: number of buckets

    Returns:
      key_func: a func with signature (arg0, arg1, arg2, src_len, tgt_len)
        and returns an tf.int64 bucket id
    """
    raise NotImplementedError()

  def create_reduce_func(self, batch_size, src_eos_id, tgt_eos_id):
    """Reduce func for tf.contrib.data.group_by_window.

    Args:
      batch_size: batch size
      src_eos_id: source eos id
      tgt_eos_id: target eos id

    Returns:
      reduce_func: a func with signature (arg0, windowed_data) and returns
        a padded dataset.
    """
    raise NotImplementedError()


class InferIteratorHookCreator(object):
  """Create hook(s) for infer iterator."""

  def create_batching_func(self, batch_size, src_eos_id):
    """Create a batching_func with signature (x) and return a
      batch of padded dataset.

    Args:
      batch_size: batch size
      src_eos_id: eos id

    Returns:
      batching_func: a func with signature (x) and
        returns a padded batch of dataset `x`
    """
    raise NotImplementedError()


class DefaultIteratorHooksCreator(IteratorHooksCreator):
  """Default hooks creator for training and evaluation's dataset iterator."""

  def create_batching_func(self, batch_size, src_eos_id, tgt_eos_id):

    def _batching_func(x):
      return x.padded_batch(
        batch_size,
        padded_shapes=(
          tf.TensorShape([None]),  # source
          tf.TensorShape([None]),  # target input
          tf.TensorShape([None]),  # target output
          tf.TensorShape([]),  # source sequence length
          tf.TensorShape([])),  # target sequence length
        padding_values=(
          src_eos_id,  # source
          tgt_eos_id,  # target input
          tgt_eos_id,  # target output
          0,  # do not need to pad
          0))  # do not need to pad

    return _batching_func

  def create_key_func(self, src_max_len, num_buckets):

    def _key_func(arg0, arg1, arg2, src_len, tgt_len):
      if src_max_len:
        bucket_width = (src_max_len + num_buckets - 1) // num_buckets
      else:
        bucket_width = 10

      bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
      return tf.to_int64(tf.minimum(num_buckets, bucket_id))

    return _key_func

  def create_reduce_func(self, batch_size, src_eos_id, tgt_eos_id):

    def _reduce_func(arg0, windowed_data):
      batching_func = self.create_batching_func(
        batch_size, src_eos_id, tgt_eos_id)
      return batching_func(windowed_data)

    return _reduce_func


class DefaultInferIteratorHookCreator(InferIteratorHookCreator):
  """Default hook creator for inference's dataset iterator."""

  def create_batching_func(self, batch_size, src_eos_id):
    def _batching_func(x):
      return x.padded_batch(
        batch_size,
        padded_shapes=(tf.TensorShape([None]),  # source
                       tf.TensorShape([])),  # source sequence length
        padding_values=(src_eos_id,  # pad source with eos
                        0))  # do not need pad source_sequence_length

    return _batching_func
