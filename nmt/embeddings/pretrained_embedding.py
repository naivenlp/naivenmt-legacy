import codecs

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import lookup_ops

from nmt.embeddings.embedding import Embedding


class PretrainedEmbedding(Embedding):

    def __init__(self,
                 vocab_file,
                 pretrained_file,
                 name="pretrained_embedding",
                 dtype=tf.float32,
                 scope="embedding"):
        """Init.

        Args:
            vocab_file: A python string, vocab file's path
            pretrained_file: A python string, pretrained embedding file
            name: A python string, embedding variable's name
            dtype: Data type
            scope: A python string, variable scope
        """
        self.name = name
        self.dtype = dtype
        self.scope = scope
        self.vocab_file = vocab_file
        self.pretrained_file = pretrained_file
        self.str2id = None
        self.id2str = None

    def embedding(self, inputs, length, params=None):
        default_config = self.default_config()
        if params:
            default_config.update(**params)
        params = default_config

        # load vocab
        vocab, vocab_size = self._load_vocab()
        params.update({'vocab_size': vocab_size})

        # load pretrained embedding
        embedding_dict, embedding_size = self._load_pretrained_embedding()
        trainable_tokens = []
        for v in vocab:
            if v not in embedding_dict:
                trainable_tokens.append(v)
        num_trainable_tokens = len(trainable_tokens)

        for token in trainable_tokens:
            embedding_dict[token] = [0.0] * embedding_size

        # update vocab, move trainable tokens to the front of the vocab
        self._update_vocab(trainable_tokens)
        vocab, vocab_size = self._load_vocab()
        embedding_matrix = np.array([embedding_dict[v] for v in vocab], dtype=self.dtype.as_numpy_dtype)
        embedding_matrix = tf.convert_to_tensor(embedding_matrix)
        # vocabs in pretrained file is pretrained vectors, so it is constant
        const_matrix = tf.slice(input_=embedding_matrix, begin=[num_trainable_tokens, 0], size=[-1, -1])

        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE) as scope:
            # vocabs not is pretrained file is trainable variables
            variable_matrix = tf.get_variable(
                name=self.name,
                shape=[num_trainable_tokens, embedding_size],
                dtype=self.dtype)

        # concat trainable embeddings and constant embeddings
        embeddings = tf.concat([variable_matrix, const_matrix], axis=0)
        self.str2id = lookup_ops.index_table_from_file(self.vocab_file, default_value=params['unk_id'])
        self.id2str = lookup_ops.index_to_string_table_from_file(self.vocab_file, default_value=params['unk'])

        input_ids = self.str2id.lookup(inputs)
        embedded_inputs = tf.nn.embedding_lookup(embeddings, input_ids)
        return embedded_inputs

    def default_config(self):
        params = {
            "unk": "<unk>",
            "unk_id": 0,
            "vocab_size": 10000,
            "embedding_size": 256,
            "num_partitions": 0
        }
        return params

    def _update_vocab(self, trainable_tokens):
        """Update vocab file. Move trainable tokens to the front of the vocab.

        Args:
            trainable_tokens: A list of str, vocabs that not in pretrained file.
        """
        tokens = []
        with open(self.vocab_file, mode='rt', encoding='utf8', buffering=8192) as fin:
            for v in fin:
                v = v.strip('\n')
                if v not in set(trainable_tokens):
                    tokens.append(v)
        processed_vocab = self.vocab_file + '.processed'
        with open(processed_vocab, mode='wt', encoding='utf8', buffering=8192) as fout:
            for t in trainable_tokens:
                fout.write(t + '\n')
            for t in tokens:
                fout.write(t + '\n')
        self.vocab_file = processed_vocab

    def _load_vocab(self):
        vocabs = []
        with codecs.getreader("utf8")(tf.gfile.GFile(self.vocab_file, "rb")) as f:
            for v in f:
                vocabs.append(v.strip('\n'))
        return vocabs, len(vocabs)

    def _load_pretrained_embedding(self):
        embedding_dict = dict()
        embedding_size = None
        with codecs.getreader("utf-8")(tf.gfile.GFile(self.pretrained_file, "rb")) as f:
            for line in f:
                tokens = line.strip().split(" ")
                word = tokens[0]
                vec = list(map(float, tokens[1:]))
                embedding_dict[word] = vec
                if embedding_size:
                    assert embedding_size == len(vec), "All embedding size should be same"
                else:
                    embedding_size = len(vec)
        return embedding_dict, embedding_size
