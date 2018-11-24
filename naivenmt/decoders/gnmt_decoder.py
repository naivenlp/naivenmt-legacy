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
from tensorflow.python.util import nest

from naivenmt.decoders.attention_decoder import AttentionDecoder
from naivenmt.decoders.gnmt_attention_multi_cell import GNMTAttentionMultiCell


class GNMTDecoder(AttentionDecoder):

  def __init__(self,
               params,
               embedding,
               sos_id,
               eos_id,
               scope="gnmt_decoder",
               dtype=tf.float32):
    super(AttentionDecoder, self).__init__(
      params=params,
      embedding=embedding,
      sos_id=sos_id,
      eos_id=eos_id,
      scope=scope,
      dtype=dtype)

  def _build_decoder_cell(self,
                          mode,
                          encoder_outputs,
                          encoder_state,
                          sequence_length):
    attention_option = self.attention
    attention_architecture = self.attention_architecture
    if attention_architecture == "standard":
      return super(AttentionDecoder, self)._build_decoder_cell(
        mode, encoder_outputs, encoder_state, sequence_length)

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

    cell_list = self._create_rnn_cell(mode, residual_fn=self._residual_fn)
    attention_cell = cell_list.pop(0)

    alignment_history = (
        mode == tf.estimator.ModeKeys.PREDICT and self.beam_width == 0)
    attention_cell = tf.contrib.seq2seq.AttentionWrapper(
      attention_cell,
      attention_mechanism,
      attention_layer_size=self.num_units,
      alignment_history=alignment_history,
      output_attention=self.output_attention,
      name="attention")
    if attention_architecture == "gnmt":
      cell = GNMTAttentionMultiCell(attention_cell, cell_list)
    elif attention_architecture == "gnmt_v2":
      cell = GNMTAttentionMultiCell(attention_cell, cell_list,
                                    use_new_attention=True)
    else:
      raise ValueError(
        "Invalid attention architecture: %s" % attention_architecture)
    if self.pass_hidden_state:
      decoder_initial_state = tuple(
        zs.clone(cell_state=es)
        if isinstance(zs, tf.contrib.seq2seq.AttentionWrapperState) else es
        for zs, es in zip(
          cell.zero_state(batch_size, self.dtype), encoder_state))
    else:
      decoder_initial_state = cell.zero_state(batch_size, self.dtype)

    return cell, decoder_initial_state

  @staticmethod
  def _residual_fn(inputs, outputs):
    def split_input(inp, out):
      out_dim = out.get_shape().as_list()[-1]
      inp_dim = inp.get_shape().as_list()[-1]
      return tf.split(inp, [out_dim, inp_dim - out_dim], axis=-1)

    actual_inputs, _ = nest.map_structure(split_input, inputs, outputs)

    def assert_shape_match(inp, out):
      inp.get_shape().assert_is_compatible_with(out.get_shape())

    nest.assert_same_structure(actual_inputs, outputs)
    nest.map_structure(assert_shape_match, actual_inputs, outputs)
    return nest.map_structure(lambda inp, out: inp + out, actual_inputs,
                              outputs)
