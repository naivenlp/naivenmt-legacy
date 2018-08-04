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
               sos,
               eos,
               scope=None,
               dtype=None,
               single_cell_fn=None):
    super().__init__(params=params,
                     embedding=embedding,
                     sos=sos,
                     eos=eos,
                     scope=scope,
                     dtype=dtype)

    assert params.attention is None

    self.unit_type = params.unit_type
    self.num_units = params.num_units
    self.num_decoder_layers = params.num_decoder_layers
    self.num_decoder_residual_layers = params.num_decoder_residual_layers
    self.forget_bias = params.forget_bias
    self.dropout = params.dropout
    self.num_gpus = params.num_gpus
    self.beam_width = params.beam_width

    self.single_cell_fn = single_cell_fn
    if not self.single_cell_fn:
      self.single_cell_fn = self._single_cell_fn

  def _build_decoder_cell(self,
                          mode,
                          encoder_outputs,
                          encoder_state,
                          sequence_length):
    cell = self._create_rnn_cell(mode)
    if mode == tf.estimator.ModeKeys.PREDICT:
      decoder_initial_state = tf.contrib.seq2seq.tile_batch(
        encoder_state, multiplier=self.beam_width)
    else:
      decoder_initial_state = encoder_state
    return cell, decoder_initial_state

  def _create_rnn_cell(self, mode):
    cell_list = self._cell_list(
      self.unit_type,
      self.num_units,
      self.num_decoder_layers,
      self.num_decoder_residual_layers,
      self.forget_bias,
      self.dropout,
      mode=mode,
      num_gpus=self.num_gpus,
      single_cell_fn=self.single_cell_fn)
    if len(cell_list) == 1:
      return cell_list[0]
    else:
      return tf.contrib.rnn.MultiRNNCell(cell_list)

  def _cell_list(self,
                 unit_type,
                 num_units,
                 num_layers,
                 num_residual_layers,
                 forget_bias,
                 dropout,
                 mode,
                 num_gpus,
                 base_gpu=0,
                 single_cell_fn=None,
                 residual_fn=None):
    cells = []
    assert single_cell_fn
    for i in range(num_layers):
      residual_conn = (i >= num_layers - num_residual_layers)
      device_str = self._get_device_str(i + base_gpu, num_gpus)
      single_cell = single_cell_fn(
        unit_type, num_units, forget_bias, dropout, mode, residual_conn,
        device_str, residual_fn)
      cells.append(single_cell)
    return cells

  @staticmethod
  def _single_cell_fn(unit_type,
                      num_units,
                      forget_bias,
                      dropout, mode,
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
