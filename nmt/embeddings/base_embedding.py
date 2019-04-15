from nmt.embeddings.embedding import Embedding
import tensorflow as tf
import codecs
from tensorflow.python.ops import lookup_ops


class BaseEmbedding(Embedding):

    def __init__(self, vocab_file, dtype=tf.float32, name="base_embedding", scope="embedding"):
        self.dtype = dtype
        self.name = name
        self.scope = scope

        self.vocab_file = vocab_file
        self.str2id = None
        self.id2str = None

    def embedding(self, inputs, length, params=None):
        default_params = self.default_config()
        if params:
            default_params.update(**params)
        params = default_params

        self._check_vocab(params)

        # create look up tables
        self.str2id = lookup_ops.index_table_from_file(self.vocab_file, default_value=params['unk_id'])
        self.id2str = lookup_ops.index_to_string_table_from_file(self.vocab_file, default_value=params['unk'])

        return self._embedding(inputs, length, params)

    def _embedding(self, inputs, length, params):
        raise NotImplementedError()

    def _check_vocab(self, params):
        vocab_size = 0
        with codecs.getreader("utf-8")(tf.gfile.GFile(self.vocab_file, "rb")) as f:
            for word in f:
                vocab_size += 1
        params['vocab_size'] = vocab_size

    def default_config(self):
        params = {
            "unk": "<unk>",
            "unk_id": 0,
            "vocab_size": 10000,
            "embedding_size": 256,
            "num_partitions": 0
        }
        return params
