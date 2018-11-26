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


class EncoderInterface(abc.ABC):
  """Encoder interface."""

  @abc.abstractmethod
  def encode(self, mode, sequence_inputs, sequence_length):
    """Encode source inputs.

    Args:
      mode: mode
      sequence_inputs: A tensor, embedding representation of inputs sequence
      sequence_length: A tensor, input sequences' length

    Returns:
      encoder_outputs: A tensor, outputs of encoder
      encoder_state: A tensor, states of encoder
    """
    raise NotImplementedError()


class AbstractEncoder(EncoderInterface):
  """Abstract encoder."""

  def __init__(self,
               params,
               scope="encoder",
               dtype=tf.float32):
    """Init abstract encoder.

    Args:
      params: A python object, hparams
      scope: A constant string, variables scope
      dtype: A constant, variables dtype
    """

    self.scope = scope
    self.dtype = dtype

    self.num_encoder_layers = params.num_encoder_layers
    self.num_encoder_residual_layers = params.num_encoder_residual_layers
    self.encoder_type = params.encoder_type
    self.time_major = params.time_major
    self.unit_type = params.unit_type
    self.num_units = params.num_units
    self.forget_bias = params.forget_bias
    self.dropout = params.dropout

  def encode(self, mode, sequence_inputs, sequence_length):
    num_layers = self.num_encoder_layers
    num_residual_layers = self.num_encoder_residual_layers

    with tf.variable_scope(self.scope, dtype=self.dtype):
      if self.time_major:
        sequence_inputs = tf.transpose(sequence_inputs, perm=[1, 0, 2])

      if self.encoder_type == "uni":
        cell = self._build_encoder_cell(mode, num_layers, num_residual_layers)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
          cell=cell,
          inputs=sequence_inputs,
          dtype=self.dtype,
          sequence_length=sequence_length,
          time_major=self.time_major,
          swap_memory=True)
      elif self.encoder_type == "bi":
        num_bi_layers = int(num_layers / 2)
        num_bi_residual_layers = int(num_residual_layers / 2)

        fw_cell, bw_cell = self._build_bidirectional_encoder_cell(
          mode=mode,
          num_bi_layers=num_bi_layers,
          num_bi_residual_layers=num_bi_residual_layers)
        bi_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
          cell_fw=fw_cell,
          cell_bw=bw_cell,
          inputs=sequence_inputs,
          dtype=self.dtype,
          sequence_length=sequence_length,
          time_major=self.time_major,
          swap_memory=True)
        encoder_outputs = tf.concat(bi_outputs, -1)
        if num_bi_layers == 1:
          encoder_state = bi_encoder_state
        else:
          encoder_state = []
          for layer_id in range(num_bi_layers):
            encoder_state.append(bi_encoder_state[0][layer_id])
            encoder_state.append(bi_encoder_state[1][layer_id])
          encoder_state = tuple(encoder_state)
      else:
        raise ValueError("Invalid encoder type: %s" % self.encoder_type)
      return encoder_outputs, encoder_state

  @abc.abstractmethod
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
    raise NotImplementedError()

  def _build_bidirectional_encoder_cell(self,
                                        mode,
                                        num_bi_layers,
                                        num_bi_residual_layers):
    """Create bi-directional cells.

    Args:
      mode: mode
      num_bi_layers: A integer, number of bidirectional layers
      num_bi_residual_layers: A integer, number of bidirectional residual layers

    Returns:
      Encoder's forward and backward rnn cells.
    """
    forward_cell = self._build_encoder_cell(
      mode=mode,
      num_layers=num_bi_layers,
      num_residual_layers=num_bi_residual_layers)
    backward_cell = self._build_encoder_cell(
      mode=mode,
      num_layers=num_bi_layers,
      num_residual_layers=num_bi_residual_layers)
    return forward_cell, backward_cell
