import abc
import tensorflow as tf


class Encoder(abc.ABC):

  @abc.abstractmethod
  def encode(self, embedding_input, sequence_length, params):
    raise NotImplementedError()


class DefaultEncoder(Encoder):

  def __init__(self, scope="encoder"):
    self.scope = scope

  def encode(self, embedding_input, sequence_length, params):
    with tf.variable_scope(self.scope) as scope:
      pass
