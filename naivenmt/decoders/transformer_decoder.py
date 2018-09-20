from naivenmt.decoders import DecoderInterface

import tensorflow as tf
from naivenmt.layers import transformer


class TransformerDecoder(DecoderInterface):

  def __init__(self,
               embedding,
               num_layers=6,
               model_dim=512,
               num_heads=8,
               ffn_dim=2948,
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
    self.scope = scope or "decoder"

  def decode(self, mode, encoder_outputs, encoder_state, labels, src_seq_len):
    self.dropout = self.dropout if mode == tf.estimator.ModeKeys.TRAIN else 0.0

    with tf.variable_scope(self.scope, dtype=self.dtype) as scope:
      decoder_embedding_input = self.embedding.decoder_embedding_input(labels.target_input_ids)
      decoder_embedding_input += transformer.positional_encoding(decoder_embedding_input, self.model_dim)

      padding_mask = transformer.padding_mask(decoder_embedding_input, decoder_embedding_input, self.num_heads)
      sequence_mask = transformer.sequence_mask(decoder_embedding_input, self.num_heads, self.dtype)
      self_attention_mask = tf.greater(padding_mask + sequence_mask, 0)
      context_attention_mask = transformer.padding_mask(encoder_outputs, decoder_embedding_input, self.num_heads)
      self_attentions = []
      ctx_attentions = []
      output = decoder_embedding_input
      for i in range(self.num_layers):
        output, self_attn, ctx_attn = self.decoder_layer(output, encoder_outputs, self_attention_mask,
                                                         context_attention_mask)
        self_attentions.append(self_attn)
        ctx_attentions.append(ctx_attn)
    return output, self_attentions, ctx_attentions

  def decoder_layer(self, decoder_inputs, encoder_outputs, self_attn_mask=None, ctx_attn_mask=None):
    decoder_output, self_atten = transformer.multihead_attention(decoder_inputs, decoder_inputs, decoder_inputs,
                                                                 self.num_heads, self.dropout, self_attn_mask)
    decoder_output, ctx_attn = transformer.multihead_attention(encoder_outputs, encoder_outputs, decoder_output,
                                                               self.num_heads, self.dropout, ctx_attn_mask)
    decoder_output = transformer.positional_wise_feed_forward_network(decoder_output, self.model_dim,
                                                                      self.ffn_dim, self.dropout)
    return decoder_output, self_atten, ctx_attn
