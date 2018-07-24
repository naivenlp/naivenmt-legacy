import abc


class Inputter(abc.ABC):

  def __init__(self, dtype):
    self.dtype = dtype

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
