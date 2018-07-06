import abc
import tensorflow as tf


class Inputter(abc.ABC):

    @abc.abstractmethod
    def build_dataset(self):
        raise NotImplementedError()


class TextFileInputter(Inputter):

    def __init__(self, file):
        self.file = file

    def build_dataset(self):
        return tf.data.TextLineDataset(self.file)


class WordEmbedder(TextFileInputter):

    def build_dataset(self):
        pass
