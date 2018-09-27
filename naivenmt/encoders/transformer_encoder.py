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
               dtype=None,
               scope=None):
    self.embedding = embedding
    self.num_layers = num_layers
    self.model_dim = model_dim
    self.num_heads = num_heads
    self.ffn_dim = ffn_dim
    self.dropout = dropout
    self.dtype = dtype or tf.float32
    self.scope = scope or "encoder"

  def encode(self, mode, features):
    self.dropout = self.dropout if mode == tf.estimator.ModeKeys.TRAIN else 0.0
    # max_sequence_length = max(self.max_seq_len, features.source_sequence_length)

    with tf.variable_scope(self.scope, dtype=self.dtype) as scope:
      encoder_embedding_input = self.embedding.encoder_embedding_input(
        features.source_ids)
      encoder_embedding_input += transformer.positional_encoding(
        encoder_embedding_input, self.model_dim)

      self_attention_mask = transformer.padding_mask(
        features.source_ids, features.source_ids, self.num_heads)

      attentions = []
      output = encoder_embedding_input
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
