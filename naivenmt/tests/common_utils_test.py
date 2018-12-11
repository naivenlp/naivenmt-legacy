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

from naivenmt.tests import common_test_utils as utils


class CommonUtilsTest(tf.test.TestCase):

  def testGetUniLSTMEncoderResults(self):
    for num_layers in [2, 4, 6]:
      outputs, _, states = utils.get_uni_lstm_encoder_results(num_layers)
      self.assertEqual(num_layers, len(states))

      for i in range(num_layers):
        self.assertEqual(2, len(states[0]))  # (states_c,states_h)

      states_c, states_h = [], []
      for i in range(num_layers):
        states_c.append(states[i][0])
        states_h.append(states[i][1])
      states_c = tf.convert_to_tensor(states_c)
      states_h = tf.convert_to_tensor(states_h)

      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())
        outputs, states_c, states_h = sess.run([outputs, states_c, states_h])
        self.assertAllEqual([utils.TIME_STEPS, utils.BATCH_SIZE, utils.DEPTH],
                            outputs.shape)
        print(states_c)
        print(states_h)
        self.assertAllEqual([num_layers, utils.BATCH_SIZE, utils.DEPTH],
                            states_c.shape)
        self.assertAllEqual([num_layers, utils.BATCH_SIZE, utils.DEPTH],
                            states_h.shape)

  def testGetUniLayerNormLSTMEncoderResults(self):
    for num_layers in [2, 4, 6]:
      outputs, _, states = utils.get_uni_lstm_encoder_results(num_layers)
      self.assertEqual(num_layers, len(states))

      for i in range(num_layers):
        self.assertEqual(2, len(states[0]))  # (states_c,states_h)

      states_c, states_h = [], []
      for i in range(num_layers):
        states_c.append(states[i][0])
        states_h.append(states[i][1])
      states_c = tf.convert_to_tensor(states_c)
      states_h = tf.convert_to_tensor(states_h)

      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())
        outputs, states_c, states_h = sess.run([outputs, states_c, states_h])
        self.assertAllEqual([utils.TIME_STEPS, utils.BATCH_SIZE, utils.DEPTH],
                            outputs.shape)
        print(states_c)
        print(states_h)
        self.assertAllEqual([num_layers, utils.BATCH_SIZE, utils.DEPTH],
                            states_c.shape)
        self.assertAllEqual([num_layers, utils.BATCH_SIZE, utils.DEPTH],
                            states_h.shape)

  def testGetUniNASEncoderResults(self):
    for num_layers in [2, 4, 6]:
      outputs, _, states = utils.get_uni_lstm_encoder_results(num_layers)
      self.assertEqual(num_layers, len(states))

      for i in range(num_layers):
        self.assertEqual(2, len(states[0]))  # (states_c,states_h)

      states_c, states_h = [], []
      for i in range(num_layers):
        states_c.append(states[i][0])
        states_h.append(states[i][1])
      states_c = tf.convert_to_tensor(states_c)
      states_h = tf.convert_to_tensor(states_h)

      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())
        outputs, states_c, states_h = sess.run([outputs, states_c, states_h])
        self.assertAllEqual([utils.TIME_STEPS, utils.BATCH_SIZE, utils.DEPTH],
                            outputs.shape)
        print(states_c)
        print(states_h)
        self.assertAllEqual([num_layers, utils.BATCH_SIZE, utils.DEPTH],
                            states_c.shape)
        self.assertAllEqual([num_layers, utils.BATCH_SIZE, utils.DEPTH],
                            states_h.shape)

  def testGetUniGRUEncoderResults(self):
    for num_layers in [2, 4, 6]:
      outputs, _, states = utils.get_uni_gru_encoder_results(num_layers)
      self.assertEqual(num_layers, len(states))

      states_list = []
      for i in range(num_layers):
        states_list.append(states[i])
      states = tf.convert_to_tensor(states_list)

      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())
        outputs, states = sess.run([outputs, states])
        self.assertAllEqual([utils.TIME_STEPS, utils.BATCH_SIZE, utils.DEPTH],
                            outputs.shape)
        self.assertAllEqual([num_layers, utils.BATCH_SIZE, utils.DEPTH],
                            states.shape)


if __name__ == "__main__":
  tf.test.main()
