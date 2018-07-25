import abc

import tensorflow as tf
from tensorflow.python.layers import core
from . import utils


class Decoder(abc.ABC):

  @abc.abstractmethod
  def decode(self, mode, encoder_outputs, encoder_state,
             labels, src_seq_len, tgt_seq_len, params):
    raise NotImplementedError()


class BaseDecoder(Decoder):

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


class BasicDecoder(BaseDecoder):

  def _build_decoder_cell(self,
                          mode,
                          params,
                          encoder_outputs,
                          encoder_state,
                          sequence_length):
    if params.attention:
      raise ValueError("Basic model does not support attention.")

    cell = self._create_rnn_cell(mode, params)
    if mode == tf.estimator.ModeKeys.PREDICT:
      decoder_initial_state = tf.contrib.seq2seq.tile_batch(
        encoder_state, multiplier=params.beam_width)
    else:
      decoder_initial_state = encoder_state
    return cell, decoder_initial_state

  def _create_rnn_cell(self, mode, params):
    if self.extra_args:
      single_cell_fn = self.extra_args.single_cell_fn
    else:
      single_cell_fn = None

    if not single_cell_fn:
      single_cell_fn = self._single_cell

    cell_list = []
    for i in range(params.num_layers):
      residual_conn = (i >= params.num_layers - params.num_resisual_layers)
      single_cell = single_cell_fn(
        unit_type=params.unit_type,
        num_units=params.num_units,
        forget_bias=params.forget_bias,
        dropout=params.dropout,
        mode=mode,
        residual_connection=residual_conn,
        device_str=utils.get_device_str(i, params.num_gpus))
      cell_list.append(single_cell)
    if len(cell_list) == 1:
      return cell_list[0]
    else:
      return tf.contrib.rnn.MultiRNNCell(cell_list)

  def _single_cell(self,
                   unit_type,
                   num_units,
                   forget_bias,
                   dropout,
                   mode,
                   residual_connection=False,
                   device_str=None,
                   residual_fn=None):
    dropout = dropout if mode == tf.estimator.ModeKeys.TRAIN else 0.0

    if unit_type == "lstm":
      single_cell = tf.contrib.rnn.BasicLSTMCell(
        num_units, forget_bias=forget_bias)
    elif unit_type == "gru":
      single_cell = tf.contrib.rnn.GRUCell(num_units)
    elif unit_type == "layer_norm_lstm":
      single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
        num_units, forget_bias=forget_bias, layer_norm=True)
    elif unit_type == "nas":
      single_cell = tf.contrib.rnn.NASCell(num_units)
    else:
      raise ValueError("Invalid unit type: %s" % unit_type)

    if dropout > 0.0:
      single_cell = tf.contrib.rnn.DropoutWrapper(
        cell=single_cell, input_keep_prob=(1.0 - dropout))
    if residual_connection:
      single_cell = tf.contrib.rnn.ResidualWrapper(
        single_cell, residual_fn=residual_fn)

    if device_str:
      single_cell = tf.contrib.rnn.DeviceWrapper(
        single_cell, device_str)

    return single_cell
