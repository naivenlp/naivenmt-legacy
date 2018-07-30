import abc

import tensorflow as tf
from tensorflow.python.ops import lookup_ops
from naivenmt.inputters import DefaultTrainAndEvalIteratorHook
from naivenmt.inputters import DefaultInferIteratorHook


class InputterInterface(abc.ABC):

  @abc.abstractmethod
  def iterator(self, mode):
    """Dataset's iterator"""
    raise NotImplementedError()

  @abc.abstractmethod
  def source_vocab_table(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def source_reverse_vocab_table(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def target_vocab_table(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def target_reverse_vocab_table(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def source_sequence_length(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def target_sequence_length(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def source_sequence_length_infer(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def target_sequence_length_infer(self):
    raise NotImplementedError()


class Inputter(InputterInterface):

  def __init__(self, configs, params,
               predict_file=None,
               iterator_hook=None,
               infer_iterator_hook=None):
    """Inputter for models.

    Args:
      configs: configs
      params: hparams
      predict_file: file to do predict
      iterator_hook: train and eval iterator's hook. Must be an instance of
        `naivenmt.inputters.IteratorHook` or its subclass.
      infer_iterator_hook: infer iterator's hook. Must be an instance of
        `naivenmt.inputters.InferIterator` or its subclass
    """
    self.source_train_file = configs.src_train_file
    self.target_train_file = configs.tgt_train_file
    self.source_eval_file = configs.src_dev_file
    self.target_eval_file = configs.tgt_dev_file
    self.source_vocab_file = configs.src_vocab_file
    self.target_vocab_file = configs.tgt_vocab_file
    self.batch_size = params.batch_size
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
    if not self.iterator_hook:
      self.iterator_hook = DefaultTrainAndEvalIteratorHook(
        self.batch_size,
        eos_id=self.source_vocab_table.lookup(tf.constant(self.eos)),
        src_max_len=self._source_sequence_length,
        num_buckets=self.num_buckets)
    self.infer_iterator_hook = infer_iterator_hook
    if not self.infer_iterator_hook:
      self.infer_iterator_hook = DefaultInferIteratorHook(
        batch_size=self.batch_size,
        eos_id=self.source_vocab_table.lookup(tf.constant(self.eos)))

  def iterator(self, mode):
    if mode == tf.estimator.ModeKeys.TRAIN:
      if not self._train_iterator:
        self._make_train_iterator()
      return self._train_iterator
    elif mode == tf.estimator.ModeKeys.EVAL:
      if not self._eval_iterator:
        self._make_eval_iterator()
      return self._eval_iterator
    elif mode == tf.estimator.ModeKeys.PREDICT:
      if not self._predict_iterator:
        self._make_predict_iterator()
      return self._predict_iterator
    else:
      raise ValueError("Unknown mode: %s" % mode)

  def _make_train_iterator(self):
    src_dataset = tf.data.TextLineDataset(self.source_train_file)
    tgt_dataset = tf.data.TextLineDataset(self.target_train_file)

  def _make_eval_iterator(self):
    raise NotImplementedError()

  def _make_predict_iterator(self):
    if not self._predict_file:
      raise TypeError("Inference file is NoneType.")
    raise NotImplementedError()

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
