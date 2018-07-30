import abc

import tensorflow as tf
from tensorflow.python.ops import lookup_ops

# TODO(luozhouyang) use configs
UNK_ID = 0
BATCH_SIZE = 32
SOS = "<s>"
EOS = "</s>"
RANDOM_SEED = None
NUM_BUCKETS = 5
SRC_MAX_LEN = 30
TGT_MAX_LEN = 30
SKIP_COUNT = 0
JOB_ID = 0


class InputterInterface(abc.ABC):

  @abc.abstractmethod
  def get_length(self, inputs):
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def source_sequence_length(self):
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def target_sequence_length(self):
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def target_output(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def iterator(self, mode, params):
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def serving_input_receiver(self):
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def reverse_target_vocab_table(self):
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def reverse_source_vocab_table(self):
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def source_vocab_table(self):
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def target_vocab_table(self):
    raise NotImplementedError()


class Inputter(InputterInterface):

  def __init__(self, config, dtype, mode=tf.estimator.ModeKeys.TRAIN):
    self.configs = config
    self.dtype = dtype
    self.mode = mode
    self._iterator = None
    self._source_sequence_length = None
    self._target_sequence_length = None
    self._target_output = None
    self._serving_input_receiver = None
    self._reverse_target_vocab_table = None
    self._reverse_source_vocab_table = None
    self._source_vocab_table = None
    self._target_vocab_table = None
    tf.estimator.Estimator
    self._prepare()

  def _prepare(self):
    self._source_vocab_table = lookup_ops.index_table_from_file(
      self.configs.src_vocab_file)
    self._target_vocab_table = lookup_ops.index_table_from_file(
      self.configs.tgt_vocab_file)
    self._reverse_source_vocab_table = (
      lookup_ops.index_to_string_table_from_file(self.configs.src_vocab_file))
    self._reverse_target_vocab_table = (
      lookup_ops.index_to_string_table_from_file(self.configs.tgt_vocab_file))

    if self.mode == tf.estimator.ModeKeys.TRAIN:
      src_dataset = tf.data.TextLineDataset(self.configs.src_train_file)
      tgt_dataset = tf.data.TextLineDataset(self.configs.tgt_train_file)
      self._iterator = self._create_iterator(
        src_dataset=src_dataset, tgt_dataset=tgt_dataset, batch_size=BATCH_SIZE,
        sos=SOS, eos=EOS, random_seed=RANDOM_SEED,
        num_buckets=NUM_BUCKETS, src_max_len=SRC_MAX_LEN,
        tgt_max_len=TGT_MAX_LEN,
        num_parallel_calls=4, output_buffer_size=None,
        skip_count=SKIP_COUNT, num_shards=1, shard_index=0,
        reshuffle_each_iteration=True)

  def _create_iterator(self,
                       src_dataset,
                       tgt_dataset,
                       batch_size,
                       src_max_len,
                       tgt_max_len,
                       num_buckets,
                       num_parallel_calls=4,
                       skip_count=None,
                       output_buffer_size=None,
                       num_shards=1,
                       shard_index=0,
                       reshuffle_each_iteration=True,
                       sos="<s>",
                       eos="</s>",
                       random_seed=None):
    if not output_buffer_size:
      output_buffer_size = batch_size * 1000

    src_eos_id = tf.cast(self._source_vocab_table.lookup(tf.constant(eos)),
                         tf.int32)
    tgt_sos_id = tf.cast(self._target_vocab_table.lookup(tf.constant(sos)),
                         tf.int32)
    tgt_eos_id = tf.cast(self._target_vocab_table.lookup(tf.constant(eos)),
                         tf.int32)

    src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
    src_tgt_dataset = src_tgt_dataset.shard(num_shards, shard_index)
    if skip_count is not None:
      src_tgt_dataset = src_tgt_dataset.skip(skip_count)

    src_tgt_dataset = src_tgt_dataset.shuffle(
      output_buffer_size, random_seed, reshuffle_each_iteration)
    src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (
        tf.string_split([src]).values, tf.string_split([tgt]).values),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    src_tgt_dataset = src_tgt_dataset.filter(
      lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

    if src_max_len:
      src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src[:src_max_len], tgt),
        num_parallel_calls=num_parallel_calls)
    if tgt_max_len:
      src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src, tgt[:tgt_max_len]),
        num_parallel_calls=num_parallel_calls)

    # TODO(luozhouyang) Batch inputs data

  def get_length(self, inputs):
    return len(inputs.split(" "))

  @property
  def source_sequence_length(self):
    return self._source_sequence_length

  @property
  def target_sequence_length(self):
    return self._target_sequence_length

  @property
  def target_output(self):
    return self._target_output

  def iterator(self, mode, params):
    return self._iterator

  @property
  def serving_input_receiver(self):
    return self._serving_input_receiver

  @property
  def reverse_target_vocab_table(self):
    return self._reverse_target_vocab_table

  @property
  def reverse_source_vocab_table(self):
    return self._reverse_source_vocab_table

  @property
  def source_vocab_table(self):
    return self._source_vocab_table

  @property
  def target_vocab_table(self):
    return self._target_vocab_table
