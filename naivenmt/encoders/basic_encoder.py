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

from naivenmt.encoders.abstract_encoder import AbstractEncoder


class BasicEncoder(AbstractEncoder):
  """Basic encoder."""

  def __init__(self,
               params,
               scope="basic_encoder",
               dtype=tf.float32):
    """Init encoder.

    Args:
      params: hparams
      scope: variables scope
      dtype: variables dtype
    """
    super(BasicEncoder, self).__init__(params, scope, dtype)

  def _build_encoder_cell(self,
                          mode,
                          num_layers,
                          num_residual_layers):
    """Create encoder cells.

    Args:
      mode: mode
      num_layers: A integer, number of layers
      num_residual_layers: A integer, number of residual layers

    Returns:
      Encoder's rnn cells.
    """
    cells = []
    for i in range(num_layers):
      residual = (i >= num_layers - num_residual_layers)
      cell = self._build_single_cell(
        unit_type=self.unit_type,
        num_units=self.num_units,
        forget_bias=self.forget_bias,
        dropout=self.dropout,
        mode=mode,
        residual_conn=residual,
        residual_fn=None)
      cells.append(cell)
    # cells = cells[0] if len(cells) == 1 else cells
    return tf.nn.rnn_cell.MultiRNNCell(cells)

  @staticmethod
  def _build_single_cell(unit_type,
                         num_units,
                         forget_bias,
                         dropout,
                         mode,
                         residual_conn=False,
                         residual_fn=None):
    """Build single rnn cell.

    Args:
      unit_type: A constance string, unit type
      num_units: A integer, number of rnn's units
      forget_bias: A float, forget bias for LSTM cell
      dropout: A float, dropout rate
      residual_conn: A boolean, use residual connection or not
      residual_fn: The function to map raw cell inputs and raw cell
        outputs to the actual cell outputs of the residual network.

    Returns:
      A RNNCell or it's subclass
    """
    dropout = dropout if mode != tf.estimator.ModeKeys.PREDICT else 0.0
    if unit_type == "lstm":
      single_cell = tf.nn.rnn_cell.LSTMCell(
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
      single_cell = tf.nn.rnn_cell.DropoutWrapper(single_cell, 1.0 - dropout)
    if residual_conn:
      single_cell = tf.nn.rnn_cell.ResidualWrapper(single_cell, residual_fn)
    return single_cell
