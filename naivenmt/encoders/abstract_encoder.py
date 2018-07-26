import abc

import tensorflow as tf


class EncoderInterface(abc.ABC):

  @abc.abstractmethod
  def encode(self, embedding_input, sequence_length, params, configs):
    raise NotImplementedError()


class AbstractEncoder(EncoderInterface):

  def __init__(self, embedding, scope="encoder"):
    self.embedding = embedding
    self.scope = scope

  def encode(self, embedding_input, sequence_length, params, configs):
    num_layers = params.num_encoder_layers
    num_residual_layers = params.num_encoder_residual_layers

    with tf.variable_scope(self.scope) as scope:
      pass
