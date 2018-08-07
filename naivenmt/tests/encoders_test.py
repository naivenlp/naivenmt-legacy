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
import argparse

from naivenmt.encoders import BasicEncoder
from naivenmt.configs import Hparams, add_arguments
from naivenmt.tests.common_test_utils import add_required_params
from naivenmt.embeddings import Embedding
from naivenmt.inputters import Features, Inputter


class BasicEncoderTest(tf.test.TestCase):

  def testBasicEncoder(self):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    flags, _ = parser.parse_known_args()
    add_required_params(flags=flags)
    flags.batch_size = 2
    hparams = Hparams(flags=flags).build()

    self.assertEqual(2, hparams.batch_size)

    embedding = Embedding(src_vocab_size=hparams.source_vocab_size,
                          tgt_vocab_size=hparams.target_vocab_size,
                          src_vocab_file=hparams.source_vocab_file,
                          tgt_vocab_file=hparams.target_vocab_file,
                          share_vocab=False,
                          src_embedding_size=16,
                          tgt_embedding_size=16)

    inputter = Inputter(params=hparams)

    encoder = BasicEncoder(params=hparams,
                           embedding=embedding)

    features = Features(
      source_ids=tf.constant([[46, 46, 47]]),
      source_sequence_length=tf.constant([2, 3, 1]))
    outputs, state = encoder.encode(mode=tf.estimator.ModeKeys.TRAIN,
                                    features=features)
    print("Length of state:%s " % len(state))
    self.assertEqual(hparams.num_encoder_layers, len(state))


if __name__ == "__main__":
  tf.test.main()
