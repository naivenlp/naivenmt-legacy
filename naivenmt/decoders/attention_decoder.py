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

import tensorflow as tf

from naivenmt.decoders.basic_decoder import BasicDecoder


class AttentionDecoder(BasicDecoder):
  """Standard attention decoder."""

  def __init__(self,
               params,
               embedding,
               sos_id,
               eos_id,
               scope="attention_decoder",
               dtype=tf.float32):
    super(BasicDecoder, self).__init__(
      params=params,
      embedding=embedding,
      sos_id=sos_id,
      eos_id=eos_id,
      scope=scope,
      dtype=dtype)

    assert params.attention

    self.attention = params.attention
    self.attention_architecture = params.attention_architecture
    self.output_attention = params.output_attention
    self.pass_hidden_state = params.pass_hidden_state

  def _build_decoder_cell(self,
                          mode,
                          encoder_outputs,
                          encoder_state,
                          sequence_length):
    attention_option = self.attention
    attention_architecture = self.attention_architecture
    assert attention_architecture == "standard"

    if self.time_major:
      memory = tf.transpose(encoder_outputs, [1, 0, 2])
    else:
      memory = encoder_outputs

    if mode == tf.estimator.ModeKeys.PREDICT and self.beam_width > 0:
      memory = tf.contrib.seq2seq.tile_batch(
        memory, multiplier=self.beam_width)
      sequence_length = tf.contrib.seq2seq.tile_batch(
        sequence_length, multiplier=self.beam_width)
      encoder_state = tf.contrib.seq2seq.tile_batch(
        encoder_state, multiplier=self.beam_width)
      batch_size = tf.size(sequence_length) * self.beam_width
    else:
      batch_size = tf.size(sequence_length)

    attention_mechanism = self._attention_mechanism_fn(
      attention_option, self.num_units, memory, sequence_length)

    cell = self._create_rnn_cell(mode)
    alignment_history = (
        mode == tf.estimator.ModeKeys.PREDICT and self.beam_width == 0)
    cell = tf.contrib.seq2seq.AttentionWrapper(
      cell,
      attention_mechanism,
      attention_layer_size=self.num_units,
      alignment_history=alignment_history,
      output_attention=self.output_attention,
      name="attention")

    if self.pass_hidden_state:
      decoder_initial_state = cell.zero_state(batch_size, self.dtype).clone(
        cell_state=encoder_state)
    else:
      decoder_initial_state = cell.zero_state(batch_size, self.dtype)

    return cell, decoder_initial_state

  @staticmethod
  def _attention_mechanism_fn(option,
                              num_units,
                              memory,
                              sequence_length):
    """Create attention mechanism.

    Args:
      option: A string constant, attention option
      num_units: A integer, number of units
      memory: A tensor, encoder's outputs
      sequence_length: A tensor, source sequence length
    """
    if option == "luong":
      mechanism = tf.contrib.seq2seq.LuongAttention(
        num_units, memory, memory_sequence_length=sequence_length)
    elif option == "scaled_luong":
      mechanism = tf.contrib.seq2seq.LuongAttention(
        num_units, memory, memory_sequence_length=sequence_length, scale=True)
    elif option == "bahdanau":
      mechanism = tf.contrib.seq2seq.BahdanauAttention(
        num_units, memory, memory_sequence_length=sequence_length)
    elif option == "normed_bahdanau":
      mechanism = tf.contrib.seq2seq.BahdanauAttention(
        num_units,
        memory,
        memory_sequence_length=sequence_length,
        normalize=True)
    else:
      raise ValueError("Invalid attention option: %s" % option)

    return mechanism
