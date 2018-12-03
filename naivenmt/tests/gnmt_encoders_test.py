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


class GNMTEncoderTest(tf.test.TestCase):

  def testGNMTGRUEncoder(self):
    outputs, states, _ = common_utils.get_gnmt_encode_results({
      "encoder_type": "gnmt",
      "unit_type": "lstm"
    })

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      # time major and concat bi-directional outputs
      self.assertAllEqual([2, 3, 512], tf.shape(outputs))
      # TODO(luozhouyang) what's the meaning of this shape
      self.assertAllEqual([2, 1, 3, 256], tf.shape(states))

  def testGNMTLSTMEncoder(self):
    pass

  def testGNMTLayerNormLSTMEncoder(self):
    pass

  def testGNMTNASEncoder(self):
    pass
