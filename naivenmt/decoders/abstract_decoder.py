import abc

import tensorflow as tf
from tensorflow.python.layers import core


class DecoderInterface(abc.ABC):
  """Decoder interface."""

  @abc.abstractmethod
  def decode(self, mode, encoder_outputs, encoder_state, labels, src_seq_len):
    """Decode target.

    Args:
      mode: mode
      encoder_outputs: encoder's output
      encoder_state: encoder's state
      labels: target inputs, an instance of ``naivenmt.inputters.Labels``
      src_seq_len: source sequence length

    Returns:
      logits: logits
      sample_id: sample id
      final context state: decoder's state
    """
    raise NotImplementedError()


class AbstractDecoder(DecoderInterface):

  def __init__(self,
               params,
               embedding,
               sos,
               eos,
               scope="decoder",
               dtype=tf.float32):
    """Init decoder.

    Args:
      params: hparams
      embedding: embedding, an instance of ``naivenmt.embeddings.Embedding``
      sos: sos token
      eos: eos token
      scope: variables scope
      dtype: variables dtype
    """
    self.embedding = embedding
    self.sos = sos
    self.eos = eos
    self.scope = scope
    self.dtype = dtype

    self.time_major = params.time_major
    self.beam_width = params.beam_width
    self.length_penalty_weight = params.length_penalty_weight
    self.infer_batch_size = params.infer_batch_size
    # TODO(luozhouyang) add `target_vocab_size` to hparams
    self.target_vocab_size = params.target_vocab_size
    self.tgt_max_len_infer = params.tgt_max_len_infer
    self.sampling_temperature = params.sampling_temperature
    self.random_seed = params.random_seed

  def decode(self, mode, encoder_outputs, encoder_state, labels, src_seq_len):
    tgt_seq_len = labels.target_sequence_length
    with tf.variable_scope(self.scope, dtype=self.dtype) as scope:
      cell, decoder_initial_state = self._build_decoder_cell(
        mode, encoder_outputs, encoder_state, src_seq_len)
      output_layer = core.Dense(
        self.target_vocab_size, use_bias=False, name="output_projection")

      if mode != tf.estimator.ModeKeys.PREDICT:
        helper = tf.contrib.seq2seq.TrainingHelper(
          self.embedding.decoder_embedding_input(labels),
          tgt_seq_len,
          time_major=self.time_major)
        decoder = tf.contrib.seq2seq.BasicDecoder(
          cell, helper, decoder_initial_state)
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
          decoder,
          output_time_major=self.time_major,
          swap_memory=True,
          scope=scope)
        sample_id = outputs.sample_id
        logits = output_layer(outputs.rnn_output)
      else:
        beam_width = self.beam_width
        length_penalty_weight = self.length_penalty_weight
        tgt_sos_id = self.embedding.decoder_embedding_input(
          tf.constant(self.sos))
        tgt_sos_id = tf.cast(tgt_sos_id, tf.int32)
        tgt_eos_id = self.embedding.decoder_embedding_input(
          tf.constant(self.eos))
        tgt_eos_id = tf.cast(tgt_eos_id, tf.int32)

        max_iteration = self._get_max_infer_iterations(src_seq_len)
        start_tokens = tf.fill([self.infer_batch_size], tgt_sos_id)
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
          sampling_temperature = self.sampling_temperature
          if sampling_temperature > 0.0:
            helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
              self.embedding, start_tokens, end_token,
              softmax_temperature=sampling_temperature,
              seed=self.random_seed)
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
          output_time_major=self.time_major,
          swap_memory=True,
          scope=scope)

        if beam_width > 0:
          logits = tf.no_op()
          sample_id = outputs.predicted_ids
        else:
          logits = outputs.rnn_output
          sample_id = outputs.sample_id

    return logits, sample_id, final_context_state

  @abc.abstractmethod
  def _build_decoder_cell(self,
                          mode,
                          encoder_outputs,
                          encoder_state,
                          sequence_length):
    """Build decoder cells.

    Args:
      mode: mode
      encoder_outputs: encoder's output
      encoder_state: encoder's state
      sequence_length: source sequence length
    """
    raise NotImplementedError()

  def _get_max_infer_iterations(self, sequence_length):
    if self.tgt_max_len_infer:
      max_iterations = self.tgt_max_len_infer
    else:
      decoding_length_factor = 2.0
      max_encoder_length = tf.reduce_max(sequence_length)
      max_iterations = tf.to_int32(tf.round(
        tf.to_float(max_encoder_length) * decoding_length_factor))
    return max_iterations
