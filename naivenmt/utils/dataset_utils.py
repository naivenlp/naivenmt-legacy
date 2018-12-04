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
from naivenmt.utils import collection_utils


def build_dataset(params, mode):
  if mode == tf.estimator.ModeKeys.TRAIN:
    return build_train_dataset(params)
  elif mode == tf.estimator.ModeKeys.EVAL:
    return build_eval_dataset(params)
  elif mode == tf.estimator.ModeKeys.PREDICT:
    return build_predict_dataset(params)
  else:
    raise ValueError("Invalid mode %s" % mode)


def build_train_or_eval_dataset(src_file, tgt_file, params):
  # build dataset
  src_dataset = tf.data.TextLineDataset(src_file)
  tgt_dataset = tf.data.TextLineDataset(tgt_file)
  # build dataset
  dataset = _build_dataset(
    src_dataset=src_dataset,
    tgt_dataset=tgt_dataset,
    batch_size=params.batch_size,
    sos=params.sos,
    eos=params.eos,
    random_seed=params.random_seed,
    num_buckets=params.num_buckets,
    src_max_len=params.src_max_len,
    tgt_max_len=params.tgt_max_len,
    num_parallel_calls=params.num_parallel_calls,
    buffer_size=params.buff_size,
    skip_count=params.skip_count)
  iterator = dataset.make_initializable_iterator()
  tf.add_to_collection(collection_utils.ITERATOR, iterator.initializer)
  # build (features, labels) tuple from input fn
  src, tgt_in, tgt_out, src_len, tgt_len = iterator.get_next()
  features = {
    "inputs": src,
    "inputs_length": src_len
  }
  labels = {
    "tgt_in": tgt_in,
    "tgt_out": tgt_out,
    "tgt_len": tgt_len
  }

  return features, labels


def build_train_dataset(params):
  return build_train_or_eval_dataset(
    src_file=params.source_train_file,
    tgt_file=params.target_train_file,
    params=params)


def build_eval_dataset(params):
  return build_train_or_eval_dataset(
    src_file=params.source_eval_file,
    tgt_file=params.target_eval_file,
    params=params)


def build_predict_dataset(params):
  dataset = tf.data.TextLineDataset(params.inference_input_file)
  dataset = dataset.map(lambda src: tf.string_split([src]).values)

  # we do not convert strings to ids
  # dataset = dataset.map(
  #   lambda src: tf.cast(vocab_table.lookup(src), tf.int32))

  dataset = dataset.map(lambda src: (src, tf.size(src)))

  dataset = dataset.padded_batch(
    batch_size=params.infer_batch_size,
    padded_shapes=(
      tf.TensorShape([None]),
      tf.TensorShape([])),
    padding_values=(
      params.eos,
      0))

  iterator = dataset.make_initializable_iterator()
  tf.add_to_collection(collection_utils.ITERATOR, iterator.initializer)
  src, src_len = iterator.get_next()
  features = {
    "inputs": src,
    "inputs_length": src_len
  }

  return features, None


def _build_dataset(src_dataset,
                   tgt_dataset,
                   batch_size,
                   sos,
                   eos,
                   random_seed,
                   num_buckets,
                   src_max_len,
                   tgt_max_len,
                   num_parallel_calls=4,
                   buffer_size=None,
                   skip_count=None,
                   num_shards=1,
                   shard_index=0,
                   reshuffle_each_iteration=True):
  if not buffer_size:
    buffer_size = batch_size * 1000

  dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
  dataset = dataset.shard(num_shards, shard_index)

  if skip_count:
    dataset = dataset.skip(skip_count)

  dataset = dataset.shuffle(
    buffer_size=buffer_size,
    seed=random_seed,
    reshuffle_each_iteration=reshuffle_each_iteration)

  dataset = dataset.map(
    lambda src, tgt: (
      tf.string_split([src]).values, tf.string_split([tgt]).values),
    num_parallel_calls=num_parallel_calls).prefetch(buffer_size)

  dataset = dataset.filter(
    lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

  if src_max_len:
    dataset = dataset.map(
      lambda src, tgt: (src[:src_max_len], tgt),
      num_parallel_calls=num_parallel_calls).prefetch(buffer_size)
  if tgt_max_len:
    dataset = dataset.map(
      lambda src, tgt: (src, tgt[:tgt_max_len]),
      num_parallel_calls=num_parallel_calls).prefetch(buffer_size)

  # we do not convert strings to ids
  # dataset = dataset.map(
  #   lambda src, tgt: (
  #     tf.cast(src_vocab_table.lookup(src), tf.int32),
  #     tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
  #   num_parallel_calls=num_parallel_calls).prefetch(buffer_size)

  dataset = dataset.map(
    lambda src, tgt: (src,
                      tf.concat(([sos], tgt), 0),
                      tf.concat((tgt, [eos]), 0)),
    num_parallel_calls=num_parallel_calls).prefetch(buffer_size)

  dataset = dataset.map(
    lambda src, tgt_in, tgt_out: (
      src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_out)),
    num_parallel_calls=num_parallel_calls).prefetch(buffer_size)

  def batching_func(ds):
    return ds.padded_batch(
      batch_size=batch_size,
      padded_shapes=(
        tf.TensorShape([None]),
        tf.TensorShape([None]),
        tf.TensorShape([None]),
        tf.TensorShape([]),
        tf.TensorShape([])),
      padding_values=(
        eos,
        sos,
        eos,
        0,
        0))

  if num_buckets > 1:
    def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
      if src_max_len:
        bucket_width = (src_max_len + num_buckets - 1) // num_buckets
      else:
        bucket_width = 10

      bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
      return tf.to_int64(tf.minimum(num_buckets, bucket_id))

    def reduce_func(unused_key, windowed_data):
      return batching_func(windowed_data)

    batched_dataset = dataset.apply(
      tf.contrib.data.group_by_window(
        key_func=key_func, reduce_func=reduce_func, window_size=batch_size))
  else:
    batched_dataset = batching_func(dataset)

  return batched_dataset
