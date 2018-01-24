import os
import unittest

from . import data_utils


class TestDataUtils(unittest.TestCase):

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


if __name__ == "__main__":
    unittest.main()
