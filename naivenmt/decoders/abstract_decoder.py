import abc

import tensorflow as tf
from tensorflow.python.layers import core


class DecoderInterface(abc.ABC):

  @abc.abstractmethod
  def decode(self, mode, encoder_outputs, encoder_state,
             labels, src_seq_len, tgt_seq_len, params):
    raise NotImplementedError()


class AbstractDecoder(DecoderInterface):

  def __init__(self, embedding, scope="decoder", extra_args=None):
    self.embedding = embedding
    self.scope = scope
    self.extra_args = extra_args

  def decode(self, mode, encoder_outputs, encoder_state,
             labels, src_seq_len, tgt_seq_len, params):
    with tf.variable_scope(self.scope) as scope:
      cell, decoder_initial_state = self._build_decoder_cell(
        mode, params, encoder_outputs, encoder_state, src_seq_len)
      # TODO(luozhouyang) Add `target_vocab_size` to Hparams
      output_layer = core.Dense(
        params.target_vocab_size, use_bias=False, name="output_projection")

      if mode != tf.estimator.ModeKeys.PREDICT:
        helper = tf.contrib.seq2seq.TrainingHelper(
          self.embedding.decoder_embedding_input(labels),
          tgt_seq_len,
          time_major=params.time_major)
        decoder = tf.contrib.seq2seq.BasicDecoder(
          cell, helper, decoder_initial_state)
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
          decoder,
          output_time_major=params.time_major,
          swap_memory=True,
          scope=scope)
        sample_id = outputs.sample_id
        logits = output_layer(outputs.rnn_output)
      else:
        beam_width = params.beam_width
        length_penalty_weight = params.length_penalty_weight
        tgt_sos_id = self.embedding.decoder_embedding_input(
          tf.constant(params.sos))
        tgt_sos_id = tf.cast(tgt_sos_id, tf.int32)
        tgt_eos_id = self.embedding.decoder_embedding_input(
          tf.constant(params.eos))
        tgt_eos_id = tf.cast(tgt_eos_id, tf.int32)

        max_iteration = self._get_max_infer_iterations(src_seq_len, params)
        start_tokens = tf.fill([params.batch_size], tgt_sos_id)
        end_token = tgt_eos_id

        if beam_width > 0:
          decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=cell,
            embedding=self.embedding,
            start_tokens=start_tokens,
            end_token=end_token,
            initial_state=decoder_initial_state,
            beam_width=beam_width,
            output_layer=output_layer,
            length_penalty_weight=length_penalty_weight)
        else:
          sampling_temperature = params.sampling_temperature
          if sampling_temperature > 0.0:
            helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
              self.embedding, start_tokens, end_token,
              softmax_temperature=sampling_temperature,
              seed=params.random_seed)
          else:
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
              self.embedding, start_tokens, end_token)

          decoder = tf.contrib.seq2seq.BasicDecoder(
            cell,
            helper,
            decoder_initial_state,
            output_layer=output_layer)

        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
          decoder,
          max_iterations=max_iteration,
          output_time_major=params.time_major,
          swap_memory=True,
          scope=scope)

        if beam_width > 0:
          logits = tf.no_op()
          sample_id = outputs.predicted_ids
        else:
          logits = outputs.rnn_output
          sample_id = outputs.sample_id

    return logits, sample_id, final_context_state

  def _build_decoder_cell(self,
                          mode,
                          params,
                          encoder_outputs,
                          encoder_state,
                          sequence_length):
    raise NotImplementedError()

  @staticmethod
  def _get_max_infer_iterations(sequence_length, params):
    if params.tgt_max_len_infer:
      max_iterations = params.tgt_max_len_infer
    else:
      decoding_length_factor = 2.0
      max_encoder_length = tf.reduce_max(sequence_length)
      max_iterations = tf.to_int32(tf.round(
        tf.to_float(max_encoder_length) * decoding_length_factor))
    return max_iterations
