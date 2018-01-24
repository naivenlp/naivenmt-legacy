import tensorflow as tf
import tensorflow.contrib.keras.api.keras.backend as K

from .utils import vocab_utils


class Dataset(object):

    def src_vocab_table(self):
        raise NotImplementedError("Not implemented yet.")

    def tgt_vocab_table(self):
        raise NotImplementedError("Not implemented yet.")

    def batched_dataset(self):
        raise NotImplementedError("Not implemented yet.")

    def batched_iterator(self):
        raise NotImplementedError("Not implemented yet.")

    def batched_input(self):
        raise NotImplementedError("Not implemented yet.")


class TrainDataset(Dataset):

    def src_vocab_table(self):
        pass

    def tgt_vocab_table(self):
        pass

    def batched_dataset(self):
        pass

    def batched_iterator(self):
        pass

    def batched_input(self):
        pass


class InferDataset(Dataset):

    def __init__(self,
                 src_vocab_file,
                 tgt_vocab_file,
                 share_vocab,
                 src_file,
                 tgt_file,
                 batch_size,
                 eos,
                 src_max_len=None):
        self.src_vocab_table, self.tgt_vocab_table = vocab_utils.create_vocab_table(
            src_vocab_file, tgt_vocab_file, share_vocab)
        self.src_eos_id = tf.cast(self.src_vocab_table.lookup(tf.constant(eos)), tf.int32)

        self.src_placeholder = K.placeholder(shape=[None], dtype=tf.string)
        self.batch_size_placeholder = K.placeholder(shape=[], dtype=tf.int64)
        src_dataset = tf.data.Dataset.from_tensor_slices(self.src_placeholder)
        src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values)
        if src_max_len:
            src_dataset = src_dataset.map(lambda src: src[:src_max_len])
        src_dataset = src_dataset.map(lambda src: tf.cast(self.src_vocab_table.lookup(src), tf.int32))

    def src_vocab_table(self):
        return self.src_vocab_table

    def tgt_vocab_table(self):
        return self.tgt_vocab_table

    def batched_dataset(self):
        pass

    def batched_iterator(self):
        pass

    def batched_input(self):
        pass
