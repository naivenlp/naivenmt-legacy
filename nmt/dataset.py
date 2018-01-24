import collections

import tensorflow as tf


class BatchedInput(
    collections.namedtuple(
        "BatchedInput",
        ("initializer",
         "source",
         "target_input",
         "target_output",
         "source_sequence_length",
         "target_sequence_length"))):
    pass


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
                 src_dataset,
                 src_vocab_table,
                 batch_size,
                 eos,
                 src_max_len=None):
        self.src_vocab_table = src_vocab_table
        self.src_eos_id = tf.cast(self.src_vocab_table.lookup(tf.constant(eos)), tf.int32)

        src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values)
        if src_max_len:
            src_dataset = src_dataset.map(lambda src: src[:src_max_len])
        src_dataset = src_dataset.map(lambda src: tf.cast(self.src_vocab_table.lookup(src), tf.int32))
        src_dataset = src_dataset.map(lambda src: (src, tf.size(src)))

        def batching_func(ds):
            return ds.padded_batch(
                batch_size=batch_size,
                padded_shapes=(
                    tf.TensorShape([None]),
                    tf.TensorShape([])
                ),
                padding_values=(
                    self.src_eos_id,
                    0
                )
            )

        self.batched_dataset = batching_func(src_dataset)
        self.batched_iterator = self.batched_dataset.make_initializable_iterator()

    def src_vocab_table(self):
        return self.src_vocab_table

    def tgt_vocab_table(self):
        return self.tgt_vocab_table

    def batched_dataset(self):
        return self.batched_dataset

    def batched_iterator(self):
        return self.batched_iterator

    def batched_input(self):
        src_ids, src_seq_len = self.batched_iterator.get_next()
        return BatchedInput(
            initializer=self.batched_iterator.initializer,
            source=src_ids,
            target_input=None,
            target_output=None,
            source_sequence_length=src_seq_len,
            target_sequence_length=None
        )
