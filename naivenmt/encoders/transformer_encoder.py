import tensorflow as tf

from naivenmt.encoders import EncoderInterface
from naivenmt.layers import transformer


class TransformerEncoder(EncoderInterface):

  def __init__(self,
               embedding,
               num_layers,
               model_dim=512,
               num_heads=8,
               ffn_dim=2048,
               dropout=0.2,
               dtype=tf.float32,
               scope="transformer-encoder"):
    """Init.

    Args:
      embedding: word embedding layer
      num_layers: A python integer, number of encoder layers
      model_dim: A python integer, model dimension, the same as word embedding's dimension
      num_heads: A python integer, number of heads
      ffn_dim: A python integer, dimension of feed forward network
      dropout: A python float, dropout rate
      dtype: Data type, default is `tf.float32`
      scope: A python string, variable scope
    """
    self.embedding = embedding
    self.num_layers = num_layers
    self.model_dim = model_dim
    self.num_heads = num_heads
    self.ffn_dim = ffn_dim
    self.dropout = dropout
    self.dtype = dtype
    self.scope = scope

  def encode(self, mode, sequence_inputs, sequence_length):
    """Encode module.

    Args:
      mode: A python string, one of tf.estimator.ModeKeys
      sequence_inputs: A tensor, input sequence, shape is [B, T, D]
      sequence_length: A tensor, length of input, shape is [B]

    Returns:
      A output tensor and attentions list.
    """
    self.dropout = self.dropout if mode == tf.estimator.ModeKeys.TRAIN else 0.0
    # max_sequence_length = max(self.max_seq_len, features.source_sequence_length)

    with tf.variable_scope(self.scope, dtype=self.dtype) as scope:
      # word embedding
      # encoder_embedding_input = self.embedding.encoder_embedding_input(sequence_inputs)
      # positional embedding
      sequence_inputs += transformer.positional_encoding(
        sequence_inputs, self.model_dim)

      self_attention_mask = transformer.padding_mask(sequence_inputs, sequence_inputs, self.num_heads)

      attentions = []
      output = sequence_inputs
      for i in range(self.num_layers):
        output, attention = self.encoder_layer(output, self_attention_mask)
        attentions.append(attention)
    return output, attentions

  def encoder_layer(self, inputs, attention_mask):
    output, attention = transformer.multihead_attention(
      inputs, inputs, inputs, self.num_heads, self.dropout, attention_mask)

    output = transformer.positional_wise_feed_forward_network(
      output, self.model_dim, self.ffn_dim, self.dropout)

    return output, attention
