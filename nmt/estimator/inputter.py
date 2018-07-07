import abc

import tensorflow as tf


class Inputter(abc.ABC):

  def __init__(self, dtype=None):
    self.dtype = dtype

  @abc.abstractmethod
  def build_dataset(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_serving_input_receiver(self):
    raise NotImplementedError()


class TextFileInputter(Inputter):

  def __init__(self, file, dtype=None):
    super().__init__(dtype)
    self.file = file

  def build_dataset(self):
    return tf.data.TextLineDataset(self.file)

  def get_serving_input_receiver(self):
    return None


class WordEmbedder(TextFileInputter):

  def build_dataset(self):
    pass
