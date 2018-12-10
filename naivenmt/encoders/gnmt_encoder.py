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

from naivenmt.encoders.basic_encoder import BasicEncoder


class GNMTEncoder(BasicEncoder):
  """GNMT encoder."""

  def __init__(self,
               params,
               scope="gnmt_encoder",
               dtype=tf.float32):
    """Init encoder.

    Args:
      params: hparams
      scope: variables scope
      dtype: variables dtype
    """
    super(GNMTEncoder, self).__init__(params, scope, dtype)

  def encode(self, mode, sequence_inputs, sequence_length):
    if self.encoder_type in ["uni", "bi"]:
      raise ValueError("uni or bi encoder type only support BasicEncoder.")

    if self.encoder_type != "gnmt":
      raise ValueError("Invalid encoder type: %s" % self.encoder_type)

    # GNMT only has one bidirectional layer
    num_bi_layers = 1
    num_uni_layers = self.num_encoder_layers - num_bi_layers

    with tf.variable_scope(self.scope, dtype=self.dtype, reuse=tf.AUTO_REUSE):
      if self.time_major:
        sequence_inputs = tf.transpose(sequence_inputs, perm=[1, 0, 2])

      # build bidirectional layer
      cell_fw, cell_bw = self._build_bidirectional_encoder_cell(
        mode=mode,
        num_bi_layers=num_bi_layers,
        num_bi_residual_layers=0)
      bi_encoder_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=cell_fw,
        cell_bw=cell_bw,
        inputs=sequence_inputs,
        dtype=self.dtype,
        sequence_length=sequence_length,
        time_major=self.time_major,
        swap_memory=True)
      bi_encoder_outputs = tf.concat(bi_encoder_outputs, -1)
      # bw states shape(lstm layer_norm_lstm, nas): (states_c, states_h)
      encoder_states_bw = bi_encoder_state[1]

      # build unidirectional layers
      uni_cell = self._build_encoder_cell(
        mode=mode,
        num_layers=num_uni_layers,
        num_residual_layers=self.num_encoder_residual_layers)

      encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
        cell=uni_cell,
        inputs=bi_encoder_outputs,
        dtype=self.dtype,
        sequence_length=sequence_length,
        time_major=self.time_major)

      if num_uni_layers == 1:
        # shape: ((encoder_states_bw,), (encoder_states,))
        encoder_state = (encoder_states_bw,) + (encoder_state,)

    return encoder_outputs, encoder_state
