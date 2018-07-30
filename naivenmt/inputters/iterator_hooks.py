import tensorflow as tf


class IteratorHook(object):

  def batching_func(self, x):
    raise NotImplementedError()

  def key_func(self, arg0, arg1, arg2, src_len, tgt_len):
    raise NotImplementedError()

  def reduce_func(self, arg0, windowed_data):
    raise NotImplementedError()


class InferIteratorHook(object):

  def batching_func(self, x):
    raise NotImplementedError()


class DefaultTrainAndEvalIteratorHook(IteratorHook):

  def __init__(self, batch_size, eos_id,
               src_max_len, num_buckets):
    self.batch_size = batch_size
    self.eos_id = eos_id
    self.src_max_len = src_max_len
    self.num_buckets = num_buckets

  def batching_func(self, x):
    return x.padded_batch(
      self.batch_size,
      padded_shapes=(
        tf.TensorShape([None]),  # source
        tf.TensorShape([None]),  # target input
        tf.TensorShape([None]),  # target output
        tf.TensorShape([]),  # source sequence length
        tf.TensorShape([])),  # target sequence length
      padding_values=(
        self.eos_id,
        self.eos_id,
        self.eos_id,
        0,  # do not need to pad
        0))  # do not need to pad

  def key_func(self, arg0, arg1, arg2, src_len, tgt_len):
    if self.src_max_len:
      bucket_width = (self.src_max_len + self.num_buckets - 1)
      bucket_width = bucket_width // self.num_buckets
    else:
      bucket_width = 10

    bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
    return tf.to_int64(tf.minimum(self.num_buckets, bucket_id))

  def reduce_func(self, arg0, windowed_data):
    return self.batching_func(windowed_data)


class DefaultInferIteratorHook(InferIteratorHook):

  def __init__(self, batch_size, eos_id):
    self.batch_size = batch_size
    self.src_eos_id = eos_id

  def batching_func(self, x):
    return x.padded_batch(
      self.batch_size,
      padded_shapes=(tf.TensorShape([None]),  # source
                     tf.TensorShape([])),  # source sequence length
      padding_values=(self.src_eos_id,  # pad source with eos
                      0))  # do not need pad source_sequence_length
