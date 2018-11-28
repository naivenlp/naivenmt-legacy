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

from naivenmt.configs import HParamsBuilder
from naivenmt.embeddings import Embedding
from naivenmt.encoders import BasicEncoder
from naivenmt.tests import common_test_utils as common_utils


class BasicEncoderTest(tf.test.TestCase):

  def getConfigs(self):
    return {
      "num_encoder_layers": 2,
      "num_encoder_residual_layers": 1,
      "encoder_type": "bi",
      "unit_type": "lstm",
      "num_units": 256,
      "forget_bias": None,
      "dropout": 0.5,
      "time_major": True,
      "embed_prefix": common_utils.get_testdata_file("test_embed"),
      "vocab_prefix": common_utils.get_testdata_file("test_embed_vocab"),
      "source_embedding_size": 4,
      "target_embedding_size": 4,
      "share_vocab": False
    }

  def getEmbedding(self, hparams):
    return Embedding(src_vocab_size=hparams.source_vocab_size,
                     tgt_vocab_size=hparams.target_vocab_size,
                     src_vocab_file=hparams.source_vocab_file,
                     tgt_vocab_file=hparams.target_vocab_file,
                     share_vocab=hparams.share_vocab,
                     src_embedding_size=hparams.source_embedding_size,
                     tgt_embedding_size=hparams.target_embedding_size)

  def testBasicBiEncoder(self):
    configs = self.getConfigs()
    hparams = HParamsBuilder(configs).build()
    embedding = self.getEmbedding(hparams)
    encoder = BasicEncoder(params=hparams)

    inputs = embedding.encoder_embedding_input(
      tf.constant([['Hello', 'World'], ['Hello', 'World'], ['Hello', '<PAD>']],
                  dtype=tf.string))

    outputs, state = encoder.encode(
      mode=tf.estimator.ModeKeys.TRAIN,
      sequence_inputs=inputs,
      sequence_length=tf.constant([2, 2, 2], dtype=tf.int32))

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      # time major and concat bi-directional outputs
      self.assertAllEqual([2, 3, 512], sess.run(tf.shape(outputs)))
      print(sess.run(tf.shape(outputs)))
      print(sess.run(tf.shape(state)))
      print(sess.run(tf.shape(state[0])))
      print(sess.run(tf.shape(state[1])))
      # output_state_forward: [B, D]
      self.assertAllEqual([1, 512], sess.run(tf.shape(state[0])))
      # output_state_backward: [B, D]
      self.assertAllEqual([1, 512], sess.run(tf.shape(state[1])))
      outputs = sess.run(outputs)
      state = sess.run(state)

  def testBasicUniEncoder(self):
    configs = self.getConfigs()
    configs.update({
      "encoder_type": "uni"
    })
    hparams = HParamsBuilder(configs).build()
    embedding = self.getEmbedding(hparams)
    encoder = BasicEncoder(params=hparams)

    inputs = embedding.encoder_embedding_input(
      tf.constant([['Hello', 'World'], ['Hello', 'World'], ['Hello', '<PAD>']],
                  dtype=tf.string))

    outputs, state = encoder.encode(
      mode=tf.estimator.ModeKeys.TRAIN,
      sequence_inputs=inputs,
      sequence_length=tf.constant([2, 2, 2], dtype=tf.int32))

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      print(sess.run(tf.shape(outputs)))
      print(sess.run(tf.shape(state)))
      self.assertAllEqual([2, 3, 256], sess.run(tf.shape(state)))
      # output_state_forward: [B, D]
      self.assertAllEqual([3, 256], sess.run(tf.shape(state[0])))
      # output_state_backward: [B, D]
      self.assertAllEqual([3, 256], sess.run(tf.shape(state[1])))
      # time major: [T, B, D]
      self.assertAllEqual([2, 3, 256], sess.run(tf.shape(outputs)))


if __name__ == "__main__":
  tf.test.main()
