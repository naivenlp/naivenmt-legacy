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

from naivenmt.encoders import GNMTEncoder
from naivenmt.tests import common_test_utils as common_utils

NUM_LAYERS_2 = 2
NUM_LAYERS_4 = 4
NUM_LAYERS_6 = 6


class GNMTEncoderTest(tf.test.TestCase):
  """Test gnmt encoders."""

  def runLSTMEncoder(self, encoder, num_layers):
    """Test LSTM, LayerNormLSTM and NAS gnmt encoder. GNMT has only a single bi
    directional layer, and num_layers-1 uni layers. time_major=True

    Args:
      encoder: An instance of GNMTEncoder.
      num_layers: A integer, number of layers of decoder.

    """
    inputs_ph = tf.placeholder(
      dtype=tf.float32,
      shape=(None, common_utils.TIME_STEPS, common_utils.DEPTH))
    inputs_length_ph = tf.placeholder(dtype=tf.int32, shape=(None))

    outputs, states = encoder.encode(
      mode=tf.estimator.ModeKeys.TRAIN,
      sequence_inputs=inputs_ph,
      sequence_length=inputs_length_ph)

    num_bi_layers = 1
    num_uni_layers = num_layers - num_bi_layers

    if num_uni_layers == 1:
      # states is a tuple of (states_bi_bw, states_uni)
      # states_bi_bw is a tuple (states_bi_bw)
      # states_uni is a tuple of length num_uni_layers
      states_bi_bw, states_uni = states
      self.assertEqual(1, len(states_bi_bw))
      self.assertEqual(num_uni_layers, len(states_uni))
      # states_bi_bw[0] is a tuple of (states_c, states_h)
      self.assertEqual(2, len(states_bi_bw[0]))

      # convert states from tuple to tensor
      states_list = [states_bi_bw[0]]
      for i in range(num_uni_layers):
        states_list.append(states_uni[i])
      states = tf.convert_to_tensor(states_list)
    else:
      # states is a tuple of (states_uni) of length num_uni_layers
      states_uni = states
      self.assertEqual(num_uni_layers, len(states_uni))
      states_list = []
      for i in range(num_uni_layers):
        states_list.append(states_uni[i])
      states = tf.convert_to_tensor(states_list)

    inputs, inputs_length = common_utils.get_encoder_test_inputs()
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      outputs, states = sess.run(
        [outputs, states],
        feed_dict={
          inputs_ph: inputs,
          inputs_length_ph: inputs_length
        })

      self.assertAllEqual(
        [common_utils.TIME_STEPS, common_utils.BATCH_SIZE, common_utils.DEPTH],
        outputs.shape)

      if num_uni_layers == 1:
        self.assertEqual(num_layers, len(states))
        # 2 in second dimension means states_c and states_h
        self.assertAllEqual(
          [num_layers, 2, common_utils.BATCH_SIZE, common_utils.DEPTH],
          states.shape)
      else:
        self.assertEqual(num_uni_layers, len(states))
        self.assertAllEqual(
          [num_uni_layers, 2, common_utils.BATCH_SIZE, common_utils.DEPTH],
          states.shape)

  def runGRUEncoder(self, encoder, num_layers):
    """Test GRU gnmt encoder. time_major=True

    Args:
      encoder: A instance of GNMTEncoder.
      num_layers: A integer, number of encoder layers.
    """

    inputs_ph = tf.placeholder(
      dtype=tf.float32,
      shape=(None, common_utils.TIME_STEPS, common_utils.DEPTH))
    inputs_length_ph = tf.placeholder(dtype=tf.int32, shape=(None))

    outputs, states = encoder.encode(
      mode=tf.estimator.ModeKeys.TRAIN,
      sequence_inputs=inputs_ph,
      sequence_length=inputs_length_ph)

    num_bi_layers = 1
    num_uni_layers = num_layers - num_bi_layers

    if num_uni_layers == 1:
      states_bi_bw, states_uni = states
      # states_bi_bw = (states_bi_bw,)
      self.assertEqual(1, len(states_bi_bw))
      self.assertEqual(num_uni_layers, len(states_uni))

      # unlike lstm, whose states is a tuple of (c,h),
      # gru states has only one element
      # states_bi_bw[0] is a states tensor
      states_list = [states_bi_bw[0]]
      for i in range(num_uni_layers):
        states_list.append(states_uni[i])
      states = tf.convert_to_tensor(states_list)
    else:
      states_uni = states
      self.assertEqual(num_uni_layers, len(states_uni))
      states_list = []
      for i in range(num_uni_layers):
        states_list.append(states_uni[i])
      states = tf.convert_to_tensor(states_list)

    inputs, inputs_length = common_utils.get_encoder_test_inputs()
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      outputs, states = sess.run(
        [outputs, states],
        feed_dict={
          inputs_ph: inputs,
          inputs_length_ph: inputs_length
        })

      self.assertAllEqual(
        [common_utils.TIME_STEPS, common_utils.BATCH_SIZE, common_utils.DEPTH],
        outputs.shape)

      if num_uni_layers == 1:
        self.assertEqual(num_layers, len(states))
        self.assertAllEqual(
          [num_layers, common_utils.BATCH_SIZE, common_utils.DEPTH],
          states.shape)
      else:
        self.assertEqual(num_uni_layers, len(states))
        self.assertAllEqual(
          [num_uni_layers, common_utils.BATCH_SIZE, common_utils.DEPTH],
          states.shape)

  def testGNMTLSTMEncoder(self):
    for num_layers in [NUM_LAYERS_2, NUM_LAYERS_4, NUM_LAYERS_6]:
      configs = {
        "unit_type": "lstm",
        "encoder_type": "gnmt",
        "forget_bias": 1.0,
        "num_encoder_layers": num_layers
      }
      encoder = GNMTEncoder(params=common_utils.get_params(configs))
      self.runLSTMEncoder(encoder, num_layers)

  def testGNMTLayerNormLSTMEncoder(self):
    for num_layers in [NUM_LAYERS_2, NUM_LAYERS_4, NUM_LAYERS_6]:
      configs = {
        "unit_type": "layer_norm_lstm",
        "encoder_type": "gnmt",
        "forget_bias": 1.0,
        "num_encoder_layers": num_layers
      }
      encoder = GNMTEncoder(params=common_utils.get_params(configs))
      self.runLSTMEncoder(encoder, num_layers)

  def testGNMTNASEncoder(self):
    for num_layers in [NUM_LAYERS_2, NUM_LAYERS_4, NUM_LAYERS_6]:
      configs = {
        "unit_type": "nas",
        "encoder_type": "gnmt",
        "num_encoder_layers": num_layers
      }
      encoder = GNMTEncoder(params=common_utils.get_params(configs))
      self.runLSTMEncoder(encoder, num_layers)

  def testGNMTGRUEncoder(self):
    for num_layers in [NUM_LAYERS_2, NUM_LAYERS_4, NUM_LAYERS_6]:
      configs = {
        "unit_type": "gru",
        "encoder_type": "gnmt",
        "num_encoder_layers": num_layers
      }
      encoder = GNMTEncoder(params=common_utils.get_params(configs))
      self.runGRUEncoder(encoder, num_layers)


if __name__ == "__main__":
  tf.test.main()
