# Copyright 2018 luozhouyang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import abc

import tensorflow as tf


class DecoderInterface(abc.ABC):
  """Decoder interface."""

  @abc.abstractmethod
  def decode(self, mode, encoder_outputs, encoder_state, labels, src_seq_len):
    """Decode target.

    Args:
      mode: mode
      encoder_outputs: A tensor, encoder's output
      encoder_state: A tensor, encoder's state
      labels: A dict of tensors.
      src_seq_len: A tensor, source sequence length

    Returns:
      logits: A tensor, logits
      sample_id: A tensor, sample id
      final context state: A tensor, decoder's state
    """
    raise NotImplementedError()


class AbstractDecoder(DecoderInterface):

  def __init__(self,
               params,
               embedding,
               sos_id,
               eos_id,
               scope="decoder",
               dtype=tf.float32):
    """Init decoder.

    Args:
      params: A python object, hparams
      embedding: A tensor, target sequence's embedding
      sos_id: A constant, int64 id of SOS token
      eos_id: A constant, int64 id of EOS token
      scope: A constant string, variables scope
      dtype: A dtype, variables dtype
    """
    self.sos_id = tf.to_int32(sos_id)
    self.eos_id = tf.to_int32(eos_id)
    self.scope = scope
    self.dtype = dtype

    self.embedding = embedding

    self.time_major = params.time_major
    self.beam_width = params.beam_width
    self.length_penalty_weight = params.length_penalty_weight
    self.infer_batch_size = params.infer_batch_size
    self.target_vocab_size = params.target_vocab_size
    self.tgt_max_len_infer = params.tgt_max_len_infer
    self.sampling_temperature = params.sampling_temperature
    self.random_seed = params.random_seed

  def decode(self, mode, encoder_outputs, encoder_state, labels, src_seq_len):
    with tf.variable_scope(self.scope, dtype=self.dtype,
                           reuse=tf.AUTO_REUSE) as scope:
      cell, decoder_initial_state = self._build_decoder_cell(
        mode=mode,
        encoder_outputs=encoder_outputs,
        encoder_state=encoder_state,
        source_sequence_length=src_seq_len)
      output_layer = tf.layers.Dense(
        self.target_vocab_size, use_bias=False, name="output_projection")

      if mode != tf.estimator.ModeKeys.PREDICT:
        helper = tf.contrib.seq2seq.TrainingHelper(
          inputs=labels['tgt_in'],
          sequence_length=labels['tgt_len'],
          time_major=self.time_major)
        decoder = tf.contrib.seq2seq.BasicDecoder(
          cell=cell,
          helper=helper,
          initial_state=decoder_initial_state)
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

        max_iteration = self._get_max_infer_iterations(src_seq_len)
        start_tokens = tf.fill([self.infer_batch_size], self.sos_id)
        end_token = self.eos_id

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
              embedding=self.embedding,
              start_tokens=start_tokens,
              end_token=end_token,
              softmax_temperature=sampling_temperature,
              seed=self.random_seed)
          else:
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
              embedding=self.embedding,
              start_tokens=start_tokens,
              end_token=end_token)

          decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=cell,
            helper=helper,
            initial_state=decoder_initial_state,
            output_layer=output_layer)

        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
          decoder=decoder,
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
                          source_sequence_length):
    """Build decoder cells.

    Args:
      mode: A string constant, mode
      encoder_outputs: A tensor, encoder's output
      encoder_state: A tensor, encoder's state
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
