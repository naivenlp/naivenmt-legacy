import tensorflow as tf


class IteratorHooksCreator(object):
  """Create hooks for train and eval iterator."""

  def create_batching_func(self, batch_size, src_eos_id, tgt_eos_id):
    """Create a batching_func with signature (x) and return a
      batch of padded dataset.

    Args:
      batch_size: batch size
      src_eos_id: source eos id
      tgt_eos_id: target eos id
    """
    raise NotImplementedError()

  def create_key_func(self, src_max_len, num_buckets):
    """Decide a bucket to save tokens.

    Args:
      src_max_len: maximum source sequence length
      num_buckets: number of buckets
    """
    raise NotImplementedError()

  def create_reduce_func(self, batch_size, src_eos_id, tgt_eos_id):
    """Reduce func for tf.contrib.data.group_by_window.

    Args:
      batch_size: batch size
      src_eos_id: source eos id
      tgt_eos_id: target eos id
    """
    raise NotImplementedError()


class InferIteratorHook(object):
  """Create hook(s) for infer iterator."""

  def create_batching_func(self, batch_size, src_eos_id):
    """Create a batching_func with signature (x) and return a
      batch of padded dataset.

    Args:
      batch_size: batch size
      src_eos_id: eos id
    """
    raise NotImplementedError()


class DefaultIteratorHooksCreator(IteratorHooksCreator):

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


class DefaultInferIteratorHookCreator(InferIteratorHook):

  def create_batching_func(self, batch_size, src_eos_id):
    def _batching_func(x):
      return x.padded_batch(
        batch_size,
        padded_shapes=(tf.TensorShape([None]),  # source
                       tf.TensorShape([])),  # source sequence length
        padding_values=(src_eos_id,  # pad source with eos
                        0))  # do not need pad source_sequence_length

    return _batching_func
