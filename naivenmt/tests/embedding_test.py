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

import argparse
import os

import tensorflow as tf

from naivenmt.configs import add_arguments, Hparams
from naivenmt.embeddings import Embedding
from naivenmt.tests.hparams_test import add_required_params
from naivenmt.tests.common_test_utils import *

TEST_DATA_DIR = os.path.abspath(os.path.join(os.pardir, "../", "testdata"))


class EmbeddingTest(tf.test.TestCase):

  def setUp(self):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    self.flags, _ = parser.parse_known_args()
    add_required_params(self.flags)

  def testCreateEmbedding(self):
    self.hparams = Hparams(self.flags).build()

    print("Source vocab size: %d" % self.hparams.source_vocab_size)
    print("Target vocab size: %d" % self.hparams.target_vocab_size)
    self.assertEqual(100, self.hparams.source_vocab_size)
    self.assertEqual(100, self.hparams.target_vocab_size)

    embedding = Embedding(src_vocab_size=self.hparams.source_vocab_size,
                          tgt_vocab_size=self.hparams.target_vocab_size,
                          src_vocab_file=self.hparams.source_vocab_file,
                          tgt_vocab_file=self.hparams.target_vocab_file,
                          share_vocab=False,
                          src_embedding_size=16,
                          tgt_embedding_size=16)

    with self.test_session() as sess:
      sess.run(tf.tables_initializer())
      sess.run(tf.global_variables_initializer())

      encoder_input = embedding.encoder_embedding_input(tf.constant([0, 1]))
      self.assertEqual(2, sess.run(tf.shape(encoder_input))[0])
      self.assertEqual(16, sess.run(tf.shape(encoder_input))[1])
      print(sess.run(encoder_input))

      decoder_input = embedding.decoder_embedding_input(tf.constant([0, 1, 2]))
      self.assertEqual(3, sess.run(tf.shape(decoder_input))[0])
      self.assertEqual(16, sess.run(tf.shape(decoder_input))[1])
      print(sess.run(decoder_input))

      print(sess.run(embedding.encoder_embedding_input([0, 1, 2])))

  def testLoadFromFile(self):
    self.flags.vocab_prefix = get_testdata_file("test_embed_vocab")

    self.hparams = Hparams(self.flags).build()

    print("Source vocab size: %d" % self.hparams.source_vocab_size)
    print("Target vocab size: %d" % self.hparams.target_vocab_size)
    self.assertEqual(6, self.hparams.source_vocab_size)
    self.assertEqual(6, self.hparams.target_vocab_size)

    src_embedding_file = os.path.join(TEST_DATA_DIR, "test_embed.txt")
    tgt_embedding_file = src_embedding_file

    embedding = Embedding(src_vocab_size=self.hparams.source_vocab_size,
                          tgt_vocab_size=self.hparams.target_vocab_size,
                          src_vocab_file=self.hparams.source_vocab_file,
                          tgt_vocab_file=self.hparams.target_vocab_file,
                          src_embedding_file=src_embedding_file,
                          tgt_embedding_file=tgt_embedding_file,
                          share_vocab=False,
                          src_embedding_size=4,
                          tgt_embedding_size=4)
    with self.test_session() as sess:
      sess.run(tf.tables_initializer())
      sess.run(tf.global_variables_initializer())

      encoder_input = embedding.encoder_embedding_input(tf.constant([3, 4]))
      self.assertEqual(2, sess.run(tf.shape(encoder_input))[0])
      self.assertEqual(4, sess.run(tf.shape(encoder_input))[1])
      t = tf.constant([[1.5, 2.5, 3.5, 4.5], [1.4, 2.4, 3.4, 4.4]],
                      shape=(2, 4), dtype=tf.float32)
      self.assertAllEqual(t, sess.run(encoder_input))
      print(sess.run(encoder_input))

      decoder_input = embedding.decoder_embedding_input(tf.constant([3, 4, 5]))
      self.assertEqual(3, sess.run(tf.shape(decoder_input))[0])
      self.assertEqual(4, sess.run(tf.shape(decoder_input))[1])
      t = tf.constant(
        [[1.3, 2.3, 3.3, 4.3], [1.2, 2.2, 3.3, 4.3], [1.1, 2.1, 3.1, 4.1]])
      self.assertAllEqual(t, sess.run(decoder_input))
      print(sess.run(decoder_input))

      print(sess.run(embedding.encoder_embedding_input([0, 1, 2])))


if __name__ == "__main__":
  tf.test.main()
