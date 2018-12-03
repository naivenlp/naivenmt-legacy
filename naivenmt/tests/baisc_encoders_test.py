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

from naivenmt.tests import common_test_utils as common_utils


class BasicEncoderTest(tf.test.TestCase):

  def testBasicBiGRUEncoder(self):
    outputs, state, _ = common_utils.get_basic_encode_results({
      "unit_type": "gru",
      "encoder_type": "bi"
    })

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      # time major and concat bi-directional outputs
      self.assertAllEqual([2, 3, 512], tf.shape(outputs))
      # TODO(luozhouyang) what's the meaning of this shape
      self.assertAllEqual([2, 1, 3, 256], tf.shape(state))

  def testBasicBiLSTMEncoder(self):
    outputs, state, _ = common_utils.get_basic_encode_results({
      "unit_type": "lstm",
      "encoder_type": "bi"
    })

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      # time major and concat bi-directional outputs
      self.assertAllEqual([2, 3, 512], sess.run(tf.shape(outputs)))
      print(sess.run(tf.shape(outputs)))
      print(sess.run(tf.shape(state)))
      # TODO(luozhouyang) what's the meaning of these shapes
      self.assertAllEqual([2, 1, 2, 3, 256], tf.shape(state))
      self.assertAllEqual([1, 2, 3, 256], tf.shape(state[0]))
      self.assertAllEqual([1, 2, 3, 256], tf.shape(state[1]))

  def testBasicBiLayerNormLSTMEncoder(self):
    outputs, state, _ = common_utils.get_basic_encode_results({
      "unit_type": "layer_norm_lstm",
      "encoder_type": "bi",
      "forget_bias": 1.0
    })

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      # time major and concat bi-directional outputs
      self.assertAllEqual([2, 3, 512], sess.run(tf.shape(outputs)))
      print(sess.run(tf.shape(outputs)))
      print(sess.run(tf.shape(state)))
      # TODO(luozhouyang) what's the meaning of these shapes
      self.assertAllEqual([2, 1, 2, 3, 256], tf.shape(state))
      self.assertAllEqual([1, 2, 3, 256], tf.shape(state[0]))
      self.assertAllEqual([1, 2, 3, 256], tf.shape(state[1]))

  def testBasicBiNASEncoder(self):
    outputs, state, _ = common_utils.get_basic_encode_results({
      "unit_type": "nas",
      "encoder_type": "bi"
    })

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      # time major and concat bi-directional outputs
      self.assertAllEqual([2, 3, 512], sess.run(tf.shape(outputs)))
      print(sess.run(tf.shape(outputs)))
      print(sess.run(tf.shape(state)))
      # TODO(luozhouyang) what's the meaning of these shapes
      self.assertAllEqual([2, 1, 2, 3, 256], tf.shape(state))
      self.assertAllEqual([1, 2, 3, 256], tf.shape(state[0]))
      self.assertAllEqual([1, 2, 3, 256], tf.shape(state[1]))

  def testBasicUniGRUEncoder(self):
    outputs, state, _ = common_utils.get_basic_encode_results({
      "encoder_type": "uni",
      "unit_type": "gru"
    })

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      # time major, shape: [T, B, D]
      self.assertAllEqual([2, 3, 256], sess.run(tf.shape(outputs)))
      self.assertAllEqual([2, 3, 256], sess.run(tf.shape(state)))
      print(sess.run(tf.shape(outputs)))
      print(sess.run(tf.shape(state)))

  def testBasicUniNASEncoder(self):
    outputs, state, _ = common_utils.get_basic_encode_results({
      "encoder_type": "uni",
      "unit_type": "nas"
    })

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      # time major, shape: [T, B, D]
      self.assertAllEqual([2, 3, 256], sess.run(tf.shape(outputs)))
      self.assertAllEqual([2, 2, 3, 256], sess.run(tf.shape(state)))
      print(sess.run(tf.shape(outputs)))
      print(sess.run(tf.shape(state)))

  def testBasicUniLayerNormLSTMEncoder(self):
    outputs, state, _ = common_utils.get_basic_encode_results({
      "encoder_type": "uni",
      "unit_type": "layer_norm_lstm",
      "forget_bias": 1.0
    })

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      # time major, shape: [T, B, D]
      self.assertAllEqual([2, 3, 256], sess.run(tf.shape(outputs)))
      self.assertAllEqual([2, 2, 3, 256], sess.run(tf.shape(state)))
      print(sess.run(tf.shape(outputs)))
      print(sess.run(tf.shape(state)))

  def testBasicUniLSTMEncoder(self):
    outputs, state, _ = common_utils.get_basic_encode_results({
      "encoder_type": "uni",
      "unit_type": "lstm"
    })

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      print(sess.run(tf.shape(outputs)))
      print(sess.run(tf.shape(state)))
      self.assertAllEqual([2, 2, 3, 256], sess.run(tf.shape(state)))
      self.assertAllEqual([2, 3, 256], sess.run(tf.shape(state[0])))
      self.assertAllEqual([2, 3, 256], sess.run(tf.shape(state[1])))
      # time major: [T, B, D]
      self.assertAllEqual([2, 3, 256], sess.run(tf.shape(outputs)))


if __name__ == "__main__":
  tf.test.main()
