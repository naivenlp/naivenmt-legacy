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

from naivenmt.decoders.abstract_decoder import AbstractDecoder


class BasicDecoder(AbstractDecoder):
  """Basic decoder."""

  def __init__(self,
               params,
               embedding,
               sos_id,
               eos_id,
               scope="basic_decoder",
               dtype=tf.float32):
    super(AbstractDecoder, self).__init__(
      params=params,
      embedding=embedding,
      sos_id=sos_id,
      eos_id=eos_id,
      scope=scope,
      dtype=dtype)

    self.unit_type = params.unit_type
    self.num_units = params.num_units
    self.num_decoder_layers = params.num_decoder_layers
    self.num_decoder_residual_layers = params.num_decoder_residual_layers
    self.forget_bias = params.forget_bias
    self.dropout = params.dropout
    self.beam_width = params.beam_width
    self.infer_mode = params.infer_mode

  def _build_decoder_cell(self,
                          mode,
                          encoder_outputs,
                          encoder_state,
                          source_sequence_length):
    cells = self._create_rnn_cell(mode)
    if (mode == tf.estimator.ModeKeys.PREDICT and
        self.infer_mode == "beam_search"):
      decoder_initial_state = tf.contrib.seq2seq.tile_batch(
        encoder_state, multiplier=self.beam_width)
    else:
      decoder_initial_state = encoder_state
    return cells, decoder_initial_state

  def _create_rnn_cell(self, mode, residual_fn=None):
    cells = []
    for i in range(self.num_decoder_layers):
      res = (i >= self.num_decoder_layers - self.num_decoder_residual_layers)
      cell = self._build_single_cell(
        unit_type=self.unit_type,
        num_units=self.num_units,
        forget_bias=self.forget_bias,
        dropout=self.dropout,
        mode=mode,
        residual_connection=res,
        residual_fn=residual_fn)
      cells.append(cell)
    cells = cells[0] if len(cells) == 1 else cells
    return tf.contrib.rnn.MultiRNNCell(cells)

  @staticmethod
  def _build_single_cell(unit_type,
                         num_units,
                         forget_bias,
                         dropout,
                         mode,
                         residual_connection=False,
                         residual_fn=None):
    dropout = dropout if mode == tf.estimator.ModeKeys.TRAIN else 0.0

    if unit_type == "lstm":
      single_cell = tf.nn.rnn_cell.BasicLSTMCell(
        num_units, forget_bias=forget_bias)
    elif unit_type == "gru":
      single_cell = tf.nn.rnn_cell.GRUCell(num_units)
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
    return single_cell
