import abc
import tensorflow as tf
from tensorflow.python.layers import core


class Decoder(abc.ABC):

  @abc.abstractmethod
  def decode(self,
             mode,
             encoder_outputs,
             encoder_state,
             embedding_input,
             sequence_length,
             params):
    raise NotImplementedError()


class DefaultDecoder(Decoder):

  def __init__(self, scope="decoder"):
    self.scope = scope

  def decode(self,
             mode,
             encoder_outputs,
             encoder_state,
             embedding_input,
             sequence_length,
             params):
    with tf.variable_scope(self.scope) as scope:
      cell, decoder_initial_state = self._build_decoder_cell(
        params, encoder_outputs, encoder_state, sequence_length)
      if mode != tf.estimator.ModeKeys.PREDICT:
        helper = tf.contrib.seq2seq.TrainingHelper(
          embedding_input, sequence_length, time_major=params.time_major)
        decoder = tf.contrib.seq2seq.BasicDecoder(
          cell, helper, decoder_initial_state)
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
          decoder,
          output_time_major=params.time_major,
          swap_memory=True,
          scope=scope)
        sample_id = outputs.sample_id
        # TODO(luozhouyang) Add `target_vocab_size` to Hparams
        output_layer = core.Dense(
          params.target_vocab_size, use_bias=False, name="output_projection")
        logits = output_layer(outputs.rnn_output)
      else:
        beam_width = params.beam_width
        length_penalty_weight = params.length_penalty_weight

