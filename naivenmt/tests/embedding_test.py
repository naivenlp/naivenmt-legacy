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

import tensorflow as tf

from naivenmt.configs import add_arguments, Hparams
from naivenmt.embeddings import Embedding
from naivenmt.tests.hparams_test import add_required_params


class EmbeddingTest(tf.test.TestCase):

  def setUp(self):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    flags, _ = parser.parse_known_args()
    add_required_params(flags)
    self.hparams = Hparams(flags).build()

  def testCreateEmbedding(self):
    embedding = Embedding(src_vocab_size=self.hparams.source_vocab_size,
                          tgt_vocab_size=self.hparams.target_vocab_size,
                          src_vocab_file=self.hparams.source_vocab_file,
                          tgt_vocab_file=self.hparams.target_vocab_file,
                          share_vocab=False,
                          src_embedding_size=16,
                          tgt_embedding_size=16)

    with self.test_session() as sess:
      sess.run(tf.tables_initializer)
      print(sess.run(embedding.encoder_embedding_input([0, 1])))

  def testLoadFromFile(self):
    pass


if __name__ == "__main__":
  tf.test.main()
