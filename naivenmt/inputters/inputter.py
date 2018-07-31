import abc

import tensorflow as tf
from tensorflow.python.ops import lookup_ops

from naivenmt.inputters import DefaultInferIteratorHookCreator
from naivenmt.inputters import DefaultIteratorHooksCreator


class InputterInterface(abc.ABC):
  """An abstract class defines all the abilities a inputter should have."""

  @abc.abstractmethod
  def iterator(self, mode):
    """Dataset's iterator"""
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def source_vocab_table(self):
    """Source vocab table. Convert strings to ids."""
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def source_reverse_vocab_table(self):
    """Source reverse vocab table. Convert ids to strings."""
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def target_vocab_table(self):
    """Target vocab table. Convert strings to ids."""
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def target_reverse_vocab_table(self):
    """Target reverse vocab table. Convert strings to ids."""
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def source_sequence_length(self):
    """Maximum length of source tokens. Actually all source tokens will
      be padded to the same length."""
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def target_sequence_length(self):
    """Maximum length of target tokens. Actually all target tokens will
      be padded to the same length."""
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def source_sequence_length_infer(self):
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def target_sequence_length_infer(self):
    raise NotImplementedError()


class Inputter(InputterInterface):

  def __init__(self, configs, params,
               predict_file=None,
               iterator_hook=DefaultIteratorHooksCreator(),
               infer_iterator_hook=DefaultInferIteratorHookCreator()):
    """Inputter for models.

    Args:
      configs: configs
      params: hparams
      predict_file: file to do predict
      iterator_hook: train and eval iterator's hook. Must be an instance of
        `naivenmt.inputters.IteratorHooksCreator` or its subclass.
      infer_iterator_hook: infer iterator's hook. Must be an instance of
        `naivenmt.inputters.InferIteratorHooksCreator` or its subclass
    """
    self.source_train_file = configs.src_train_file
    self.target_train_file = configs.tgt_train_file
    self.source_eval_file = configs.src_dev_file
    self.target_eval_file = configs.tgt_dev_file
    self.source_vocab_file = configs.src_vocab_file
    self.target_vocab_file = configs.tgt_vocab_file
    self.batch_size = params.batch_size
    self.infer_batch_size = params.infer_batch_size
    self.sos = configs.sos
    self.eos = configs.eos
    self.random_seed = params.random_seed
    self.num_buckets = params.num_buckets

    # We will padding inputs, so the seq's length is constant
    self._source_sequence_length = params.src_max_len
    self._target_sequence_length = params.tgt_max_len
    self._source_sequence_length_infer = params.src_max_len_infer
    self._target_sequence_length_infer = params.tgt_max_len_infer

    self._train_iterator = None
    self._eval_iterator = None
    self._predict_file = predict_file
    self._predict_iterator = None
    self._source_vocab_table = None
    self._source_reverse_vocab_table = None
    self._target_vocab_table = None
    self._target_reverse_vocab_table = None

    self.iterator_hook = iterator_hook
    self.infer_iterator_hook = infer_iterator_hook

  def iterator(self, mode):
    if mode == tf.estimator.ModeKeys.TRAIN:
      if not self._train_iterator:
        self._train_iterator = self._make_train_iterator()
      return self._train_iterator
    elif mode == tf.estimator.ModeKeys.EVAL:
      if not self._eval_iterator:
        self._eval_iterator = self._make_eval_iterator()
      return self._eval_iterator
    elif mode == tf.estimator.ModeKeys.PREDICT:
      if not self._predict_iterator:
        self._predict_iterator = self._make_predict_iterator()
      return self._predict_iterator
    else:
      raise ValueError("Unknown mode: %s" % mode)

  def _make_train_iterator(self):
    src_dataset = tf.data.TextLineDataset(self.source_train_file)
    tgt_dataset = tf.data.TextLineDataset(self.target_train_file)
    return self._make_iterator(
      src_dataset=src_dataset,
      tgt_dataset=tgt_dataset,
      src_vocab_table=self.source_vocab_table,
      tgt_vocab_table=self.target_vocab_table,
      batch_size=self.batch_size,
      sos=self.sos,
      eos=self.eos,
      random_seed=self.random_seed,
      num_buckets=self.num_buckets,
      src_max_len=self.source_sequence_length,
      tgt_max_len=self.target_sequence_length)

  def _make_eval_iterator(self):
    src_dataset = tf.data.TextLineDataset(self.source_eval_file)
    tgt_dataset = tf.data.TextLineDataset(self.target_eval_file)
    return self._make_iterator(
      src_dataset=src_dataset,
      tgt_dataset=tgt_dataset,
      src_vocab_table=self.source_vocab_table,
      tgt_vocab_table=self.target_vocab_table,
      batch_size=self.batch_size,
      sos=self.sos,
      eos=self.eos,
      random_seed=self.random_seed,
      num_buckets=self.num_buckets,
      src_max_len=self.source_sequence_length,
      tgt_max_len=self.target_sequence_length)

  def _make_iterator(self,
                     src_dataset,
                     tgt_dataset,
                     src_vocab_table,
                     tgt_vocab_table,
                     batch_size,
                     sos,
                     eos,
                     random_seed,
                     num_buckets,
                     src_max_len,
                     tgt_max_len,
                     num_parallel_calls=4,
                     output_buffer_size=None,
                     skip_count=None,
                     num_shards=1,
                     shard_index=0,
                     reshuffle_each_iteration=True):
    if not output_buffer_size:
      output_buffer_size = batch_size * 1000
    src_eos_id = tf.cast(
      src_vocab_table.lookup(tf.constant(tf.constant(eos))), tf.int32)
    tgt_sos_id = tf.cast(
      tgt_vocab_table.lookup(tf.constant(tf.constant(sos))), tf.int32)
    tgt_eos_id = tf.cast(
      tgt_vocab_table.lookup(tf.constant(tf.constant(eos))), tf.int32)

    src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
    src_tgt_dataset = src_tgt_dataset.shard(num_shards, shard_index)
    if skip_count:
      src_tgt_dataset = src_tgt_dataset.skip(skip_count)
    src_tgt_dataset = src_tgt_dataset.shuffle(
      output_buffer_size, random_seed, reshuffle_each_iteration)

    # Split sentence to words(token list, tokens) by space
    src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (
        tf.string_split([src]).values, tf.string_split([tgt]).values),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Filter zero length tokens
    src_tgt_dataset = src_tgt_dataset.filter(
      lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

    # Slice tokens with max length
    if src_max_len:
      src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src[:src_max_len], tgt),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    if tgt_max_len:
      src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src, tgt[:tgt_max_len]),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Convert word strings to ids
    src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                        tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Add a tgt_input prefixed with sos and tgt_output suffixed with eos
    src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (src,
                        tf.concat(([tgt_sos_id], tgt), 0),
                        tf.concat((tgt, tgt_eos_id), 0)),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Add sequences' length
    src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt_in, tgt_out: (
        src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    if num_buckets > 1:
      batched_dataset = src_tgt_dataset.apply(
        tf.contrib.data.group_by_window(
          key_func=self.iterator_hook.create_key_func(src_max_len, num_buckets),
          reduce_func=self.iterator_hook.create_reduce_func(
            batch_size, src_eos_id, tgt_eos_id),
          window_size=batch_size))
    else:
      batched_dataset = self.iterator_hook.create_batching_func(
        batch_size, src_eos_id, tgt_eos_id)(src_tgt_dataset)

    batched_iterator = batched_dataset.make_initializable_iterator()
    # (src_ids, tgt_input_ids, tgt_output_ids, src_len, tgt_len) = (
    #   batched_iterator.get_next())
    return batched_iterator

  def _make_predict_iterator(self):
    if not self._predict_file:
      raise TypeError("Inference file is NoneType.")

    src_dataset = tf.data.TextLineDataset(self._predict_file)
    return self._make_infer_iterator(
      src_dataset=src_dataset,
      src_vocab_table=self.source_vocab_table,
      batch_size=self.infer_batch_size,
      eos=self.eos,
      src_max_len=self.source_sequence_length)

  def _make_infer_iterator(self,
                           src_dataset,
                           src_vocab_table,
                           batch_size,
                           eos,
                           src_max_len=None):
    src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
    src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values)

    if src_max_len:
      src_dataset = src_dataset.map(lambda src: src[:src_max_len])

    # Convert word strings to ids
    src_dataset = src_dataset.map(
      lambda src: tf.cast(src_vocab_table.lookup(src), tf.int32))

    # Add sequence length
    src_dataset = src_dataset.map(lambda src: (src, tf.size(src)))

    batched_dataset = self.infer_iterator_hook.create_batching_func(
      batch_size, src_eos_id)(src_dataset)
    batched_iterator = batched_dataset.make_initializable_iterator()
    return batched_iterator

  @property
  def source_vocab_table(self):
    if not self._source_vocab_table:
      self._create_vocab_table(self.source_vocab_file)
    return self._source_vocab_table

  @property
  def source_reverse_vocab_table(self):
    if not self._source_reverse_vocab_table:
      self._create_reverse_vocab_table(self.source_vocab_file)
    return self._source_reverse_vocab_table

  @property
  def target_vocab_table(self):
    if not self._target_vocab_table:
      self._create_vocab_table(self.target_vocab_file)
    return self._target_vocab_table

  @property
  def target_reverse_vocab_table(self):
    if not self._target_reverse_vocab_table:
      self._create_reverse_vocab_table(self.target_vocab_file)
    return self._target_reverse_vocab_table

  @staticmethod
  def _create_vocab_table(file):
    return lookup_ops.index_table_from_file(file)

  @staticmethod
  def _create_reverse_vocab_table(file):
    return lookup_ops.index_to_string_table_from_file(file)

  @property
  def source_sequence_length(self):
    return self._source_sequence_length

  @property
  def source_sequence_length_infer(self):
    return self._source_sequence_length_infer

  @property
  def target_sequence_length(self):
    return self._target_sequence_length

  @property
  def target_sequence_length_infer(self):
    return self._target_sequence_length_infer
