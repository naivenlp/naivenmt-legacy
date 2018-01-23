import unittest

from .preprocess import InferDataPreprocessor
from .preprocess import TrainDataPreprocessor


class TestTrainDataPreprocessor(unittest.TestCase):
    def testSegmenattion(self):
        p = TrainDataPreprocessor(None, None)
        p.preprocess()


class TestInferDataPreprocessor(unittest.TestCase):
    def testSegmentation(self):
        p = InferDataPreprocessor(None)
        p.preprocess()


if __name__ == "__main__":
    unittest.main()
