import os

import tensorflow as tf
from tensorflow.python.ops import lookup_ops

from . import data_utils


class DataUtilsTest(tf.test.TestCase):
    def setUp(self):
        self.src_files = [os.path.join(os.curdir, '../data/test_src_01.txt'),
                          os.path.join(os.curdir, '../data/test_src_02.txt')]
        self.tgt_files = [os.path.join(os.curdir, '../data/test_tgt_01.txt'),
                          os.path.join(os.curdir, '../data/test_tgt_02.txt')]
        self.infer_files = [os.path.join(os.curdir, '../data/test_infer_01.txt'),
                            os.path.join(os.curdir, '../data/test_infer_02.txt')]

    def testMergeFiles(self):
        output_file = os.path.join(os.curdir, '../data/test_merge.txt')
        data_utils.merge_files(self.src_files, output_file)
        self.assertIs(True, os.path.exists(output_file))
        os.remove(output_file)

    def testMergeTrainFiles(self):
        train_src_output_file = os.path.join(os.curdir, '../data/test_merge_train_src.txt')
        train_tgt_output_file = os.path.join(os.curdir, '../data/test_merge_train_tgt.txt')
        data_utils.merge_train_files(self.src_files, self.tgt_files,
                                     train_src_output_file, train_tgt_output_file)
        self.assertIs(True, os.path.exists(train_src_output_file))
        self.assertIs(True, os.path.exists(train_tgt_output_file))
        os.remove(train_src_output_file)
        os.remove(train_tgt_output_file)

    def testMergeInferFiles(self):
        infer_output_file = os.path.join(os.curdir, '../data/test_merge_infer.txt')
        data_utils.merge_infer_files(self.infer_files, infer_output_file)
        self.assertIs(True, os.path.exists(infer_output_file))
        os.remove(infer_output_file)

    def testCreateInferData(self):
        src_vocab_table = lookup_ops.index_table_from_tensor(
            tf.constant(["a", "b", "c", "eos", "sos"]))
        src_dataset = tf.data.Dataset.from_tensor_slices(
            tf.constant(["c c a", "c a", "d", "f e a g"]))
        hparams = tf.contrib.training.HParams(
            random_seed=3,
            eos="eos",
            sos="sos",
            batch_size=2,
            src_max_len_infer=3)
        table_initializer = tf.tables_initializer()
        iterator = data_utils.create_infer_data(hparams, src_dataset, src_vocab_table).batched_input()
        source = iterator.source
        seq_len = iterator.source_sequence_length
        self.assertEqual([None, None], source.shape.as_list())
        self.assertEqual([None], seq_len.shape.as_list())
        with self.test_session() as sess:
            sess.run(table_initializer)
            sess.run(iterator.initializer)

            (source_v, seq_len_v) = sess.run((source, seq_len))
            self.assertAllEqual(
                [[2, 2, 0],  # c c a
                 [2, 0, 3]],  # c a eos
                source_v)
            self.assertAllEqual([3, 2], seq_len_v)

            (source_v, seq_len_v) = sess.run((source, seq_len))
            self.assertAllEqual(
                [[-1, 3, 3],  # "d" == unknown, eos eos
                 [-1, -1, 0]],  # "f" == unknown, "e" == unknown, a
                source_v)
            self.assertAllEqual([1, 3], seq_len_v)

            with self.assertRaisesOpError("End of sequence"):
                sess.run((source, seq_len))


if __name__ == "__main__":
    tf.test.main()
