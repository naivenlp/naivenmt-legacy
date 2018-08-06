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

from naivenmt.configs.arguments import add_arguments
from naivenmt.configs.hparams import Hparams
from naivenmt.inputters import Inputter
from naivenmt.tests.hparams_test import add_required_params


class InputterTest(tf.test.TestCase):

  def testInputter(self):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    flags, _ = parser.parse_known_args()
    add_required_params(flags)
    hparams = Hparams(flags).build()
    inputter = Inputter(params=hparams)

    self.assertIsNotNone(inputter.source_vocab_table)
    self.assertIsNotNone(inputter.target_vocab_table)

    unk_id_src = tf.cast(inputter.source_vocab_table.lookup(
      tf.constant("<unk>")), tf.int32)
    sos_id_src = tf.cast(inputter.source_vocab_table.lookup(
      tf.constant("<s>")), tf.int32)
    eos_id_src = tf.cast(inputter.source_vocab_table.lookup(
      tf.constant("</s>")), tf.int32)

    unk_id_tgt = tf.cast(inputter.target_vocab_table.lookup(
      tf.constant("<unk>")), tf.int32)
    sos_id_tgt = tf.cast(inputter.target_vocab_table.lookup(
      tf.constant("<s>")), tf.int32)
    eos_id_tgt = tf.cast(inputter.target_vocab_table.lookup(
      tf.constant("</s>")), tf.int32)

    # Chosen from source vocab file in testdata
    rachel_id = tf.cast(inputter.source_vocab_table.lookup(
      tf.constant("Rachel")), tf.int32)

    # Chosen from target vocab file in testdata
    khoa_id = tf.cast(inputter.target_vocab_table.lookup(
      tf.constant("Khoa")), tf.int32)

    unk_str_src = tf.cast(
      inputter.source_reverse_vocab_table.lookup(
        tf.constant(0, dtype=tf.int64)), tf.string)
    sos_str_src = tf.cast(
      inputter.source_reverse_vocab_table.lookup(
        tf.constant(1, dtype=tf.int64)), tf.string)
    eos_str_src = tf.cast(
      inputter.source_reverse_vocab_table.lookup(
        tf.constant(2, dtype=tf.int64)), tf.string)

    unk_str_tgt = tf.cast(
      inputter.target_reverse_vocab_table.lookup(
        tf.constant(0, dtype=tf.int64)), tf.string)
    sos_str_tgt = tf.cast(
      inputter.target_reverse_vocab_table.lookup(
        tf.constant(1, dtype=tf.int64)), tf.string)
    eos_str_tgt = tf.cast(
      inputter.target_reverse_vocab_table.lookup(
        tf.constant(2, dtype=tf.int64)), tf.string)

    rachel_str_src = tf.cast(
      inputter.source_reverse_vocab_table.lookup(
        tf.constant(3, dtype=tf.int64)), tf.string)

    khos_str_tgt = tf.cast(
      inputter.target_reverse_vocab_table.lookup(
        tf.constant(3, dtype=tf.int64)), tf.string)

    with self.test_session() as sess:
      sess.run(tf.tables_initializer())
      sess.run(inputter.iterator(tf.estimator.ModeKeys.TRAIN).initializer)
      print(sess.run(unk_id_src))
      print(sess.run(sos_id_src))
      print(sess.run(eos_id_src))

      self.assertEqual(0, sess.run(unk_id_src))
      self.assertEqual(1, sess.run(sos_id_src))
      self.assertEqual(2, sess.run(eos_id_src))
      self.assertEqual(3, sess.run(rachel_id))

      self.assertEqual(0, sess.run(unk_id_tgt))
      self.assertEqual(1, sess.run(sos_id_tgt))
      self.assertEqual(2, sess.run(eos_id_tgt))
      self.assertEqual(3, sess.run(khoa_id))

      self.assertEqual('<unk>', sess.run(unk_str_src).decode("utf8"))
      self.assertEqual('<s>', sess.run(sos_str_src).decode("utf8"))
      self.assertEqual('</s>', sess.run(eos_str_src).decode("utf8"))
      self.assertEqual('Rachel', sess.run(rachel_str_src).decode("utf8"))

      self.assertEqual('<unk>', sess.run(unk_str_tgt).decode("utf8"))
      self.assertEqual('<s>', sess.run(sos_str_tgt).decode("utf8"))
      self.assertEqual('</s>', sess.run(eos_str_tgt).decode("utf8"))
      self.assertEqual('Khoa', sess.run(khos_str_tgt).decode("utf8"))


if __name__ == "__main__":
  tf.test.main()
