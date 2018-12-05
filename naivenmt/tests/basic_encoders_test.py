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
      "encoder_type": "bi",
      "num_encoder_layers": 2
    })

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      print("output shape: %s" % sess.run(tf.shape(outputs)))
      print("states shape: %s" % sess.run(tf.shape(state)))
      # time major and concat bi-directional outputs
      # [time_steps, batch_size, depth*2]
      self.assertAllEqual([2, 3, 512], tf.shape(outputs))
      # [num_layers, batch_size, depth]
      self.assertAllEqual([2, 3, 256], tf.shape(state))

    outputs, state, _ = common_utils.get_basic_encode_results({
      "unit_type": "gru",
      "encoder_type": "bi",
      "num_encoder_layers": 4
    })

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      print("output shape: %s" % sess.run(tf.shape(outputs)))
      print("states shape: %s" % sess.run(tf.shape(state)))
      # time major and concat bi-directional outputs
      # [time_steps, batch_size, depth*2]
      self.assertAllEqual([2, 3, 512], tf.shape(outputs))
      # [num_layers, batch_size, depth]
      self.assertAllEqual([4, 3, 256], tf.shape(state))

  def testBasicBiLSTMEncoder(self):
    outputs, state, _ = common_utils.get_basic_encode_results({
      "unit_type": "lstm",
      "encoder_type": "bi",
      "num_encoder_layers": 2
    })

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      print("output shape: %s" % sess.run(tf.shape(outputs)))
      print("states shape: %s" % sess.run(tf.shape(state)))
      # time major and concat bi-directional outputs
      # [time_steps, batch_size, depth*2]
      self.assertAllEqual([2, 3, 512], tf.shape(outputs))
      # [num_layers, time_steps, batch_size, depth]
      self.assertAllEqual([2, 2, 3, 256], tf.shape(state))

    outputs, state, _ = common_utils.get_basic_encode_results({
      "unit_type": "lstm",
      "encoder_type": "bi",
      "num_encoder_layers": 4
    })

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      print("output shape: %s" % sess.run(tf.shape(outputs)))
      print("states shape: %s" % sess.run(tf.shape(state)))
      # time major and concat bi-directional outputs
      # [time_steps, batch_size, depth*2]
      self.assertAllEqual([2, 3, 512], tf.shape(outputs))
      # [num_layers, time_steps, batch_size, depth]
      self.assertAllEqual([4, 2, 3, 256], tf.shape(state))

  def testBasicBiLayerNormLSTMEncoder(self):
    outputs, state, _ = common_utils.get_basic_encode_results({
      "unit_type": "layer_norm_lstm",
      "encoder_type": "bi",
      "forget_bias": 1.0,
      "num_encoder_layers": 2
    })

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      print("output shape: %s" % sess.run(tf.shape(outputs)))
      print("states shape: %s" % sess.run(tf.shape(state)))
      # time major and concat bi-directional outputs
      # [time_steps, batch_size, depth*2]
      self.assertAllEqual([2, 3, 512], tf.shape(outputs))
      # [num_layers, time_steps, batch_size, depth]
      self.assertAllEqual([2, 2, 3, 256], tf.shape(state))

    outputs, state, _ = common_utils.get_basic_encode_results({
      "unit_type": "layer_norm_lstm",
      "encoder_type": "bi",
      "forget_bias": 1.0,
      "num_encoder_layers": 4
    })

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      print("output shape: %s" % sess.run(tf.shape(outputs)))
      print("states shape: %s" % sess.run(tf.shape(state)))
      # time major and concat bi-directional outputs
      # [time_steps, batch_size, depth*2]
      self.assertAllEqual([2, 3, 512], tf.shape(outputs))
      # [num_layers, time_steps, batch_size, depth]
      self.assertAllEqual([4, 2, 3, 256], tf.shape(state))

  def testBasicBiNASEncoder(self):
    outputs, state, _ = common_utils.get_basic_encode_results({
      "unit_type": "nas",
      "encoder_type": "bi",
      "num_encoder_layers": 2
    })

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      print("output shape: %s" % sess.run(tf.shape(outputs)))
      print("states shape: %s" % sess.run(tf.shape(state)))
      # time major and concat bi-directional outputs
      # [time_steps, batch_size, depth*2]
      self.assertAllEqual([2, 3, 512], tf.shape(outputs))
      # [num_layers, time_steps, batch_size, depth]
      self.assertAllEqual([2, 2, 3, 256], tf.shape(state))

    outputs, state, _ = common_utils.get_basic_encode_results({
      "unit_type": "nas",
      "encoder_type": "bi",
      "num_encoder_layers": 4
    })

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      print("output shape: %s" % sess.run(tf.shape(outputs)))
      print("states shape: %s" % sess.run(tf.shape(state)))
      # time major and concat bi-directional outputs
      # [time_steps, batch_size, depth*2]
      self.assertAllEqual([2, 3, 512], tf.shape(outputs))
      # [num_layers, time_steps, batch_size, depth]
      self.assertAllEqual([4, 2, 3, 256], tf.shape(state))

  def testBasicUniGRUEncoder(self):
    outputs, state, _ = common_utils.get_basic_encode_results({
      "encoder_type": "uni",
      "unit_type": "gru",
      "num_encoder_layers": 2
    })

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      print("output shape: %s" % sess.run(tf.shape(outputs)))
      print("states shape: %s" % sess.run(tf.shape(state)))
      # time major, shape: [time_steps, batch_size, depth]
      self.assertAllEqual([2, 3, 256], sess.run(tf.shape(outputs)))
      # [num_layers, batch_size, depth]
      self.assertAllEqual([2, 3, 256], sess.run(tf.shape(state)))

    outputs, state, _ = common_utils.get_basic_encode_results({
      "encoder_type": "uni",
      "unit_type": "gru",
      "num_encoder_layers": 4
    })

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      print("output shape: %s" % sess.run(tf.shape(outputs)))
      print("states shape: %s" % sess.run(tf.shape(state)))
      # time major, shape: [time_steps, batch_size, depth]
      self.assertAllEqual([2, 3, 256], sess.run(tf.shape(outputs)))
      # [num_layers, batch_size, depth]
      self.assertAllEqual([4, 3, 256], sess.run(tf.shape(state)))

  def testBasicUniNASEncoder(self):
    outputs, state, _ = common_utils.get_basic_encode_results({
      "encoder_type": "uni",
      "unit_type": "nas",
      "num_encoder_layers": 2
    })

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      print("output shape: %s" % sess.run(tf.shape(outputs)))
      print("states shape: %s" % sess.run(tf.shape(state)))
      # time major, shape: [time_steps, batch_size, depth]
      self.assertAllEqual([2, 3, 256], sess.run(tf.shape(outputs)))
      # [num_layers, time_steps, batch_size, depth]
      self.assertAllEqual([2, 2, 3, 256], sess.run(tf.shape(state)))

    outputs, state, _ = common_utils.get_basic_encode_results({
      "encoder_type": "uni",
      "unit_type": "nas",
      "num_encoder_layers": 4
    })

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      print("output shape: %s" % sess.run(tf.shape(outputs)))
      print("states shape: %s" % sess.run(tf.shape(state)))
      # time major, shape: [time_steps, batch_size, depth]
      self.assertAllEqual([2, 3, 256], sess.run(tf.shape(outputs)))
      # [num_layers, time_steps, batch_size, depth]
      self.assertAllEqual([4, 2, 3, 256], sess.run(tf.shape(state)))

  def testBasicUniLayerNormLSTMEncoder(self):
    outputs, state, _ = common_utils.get_basic_encode_results({
      "encoder_type": "uni",
      "unit_type": "layer_norm_lstm",
      "forget_bias": 1.0,
      "num_encoder_layers": 2
    })

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      print("output shape: %s" % sess.run(tf.shape(outputs)))
      print("states shape: %s" % sess.run(tf.shape(state)))
      # time major, shape: [time_steps, batch_size, depth]
      self.assertAllEqual([2, 3, 256], sess.run(tf.shape(outputs)))
      # [num_layers, time_steps, batch_size, depth]
      self.assertAllEqual([2, 2, 3, 256], sess.run(tf.shape(state)))

    outputs, state, _ = common_utils.get_basic_encode_results({
      "encoder_type": "uni",
      "unit_type": "layer_norm_lstm",
      "forget_bias": 1.0,
      "num_encoder_layers": 4
    })

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      print("output shape: %s" % sess.run(tf.shape(outputs)))
      print("states shape: %s" % sess.run(tf.shape(state)))
      # time major, shape: [time_steps, batch_size, depth]
      self.assertAllEqual([2, 3, 256], sess.run(tf.shape(outputs)))
      # [num_layers, time_steps, batch_size, depth]
      self.assertAllEqual([4, 2, 3, 256], sess.run(tf.shape(state)))

  def testBasicUniLSTMEncoder(self):
    outputs, state, _ = common_utils.get_basic_encode_results({
      "encoder_type": "uni",
      "unit_type": "lstm",
      "forget_bias": 1.0,
      "num_encoder_layers": 2
    })

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      print("output shape: %s" % sess.run(tf.shape(outputs)))
      print("states shape: %s" % sess.run(tf.shape(state)))
      # time major, shape: [time_steps, batch_size, depth]
      self.assertAllEqual([2, 3, 256], sess.run(tf.shape(outputs)))
      # [num_layers, time_steps, batch_size, depth]
      self.assertAllEqual([2, 2, 3, 256], sess.run(tf.shape(state)))

    outputs, state, _ = common_utils.get_basic_encode_results({
      "encoder_type": "uni",
      "unit_type": "lstm",
      "forget_bias": 1.0,
      "num_encoder_layers": 4
    })

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      print("output shape: %s" % sess.run(tf.shape(outputs)))
      print("states shape: %s" % sess.run(tf.shape(state)))
      # time major, shape: [time_steps, batch_size, depth]
      self.assertAllEqual([2, 3, 256], sess.run(tf.shape(outputs)))
      # [num_layers, time_steps, batch_size, depth]
      self.assertAllEqual([4, 2, 3, 256], sess.run(tf.shape(state)))


if __name__ == "__main__":
  tf.test.main()
