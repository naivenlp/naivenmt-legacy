from nmt.embeddings.base_embedding import BaseEmbedding
import tensorflow as tf
import numpy as np
import codecs


class PretrainedEmbedding(BaseEmbedding):

    def __init__(self,
                 vocab_file,
                 pretrained_file,
                 num_special_tokens=3,
                 name="pretrained_embedding",
                 dtype=tf.float32,
                 scope="embedding"):
        """Init.

        Args:
            vocab_file: A python string, vocab file's path
            pretrained_file: A python string, pretrained embedding file
            num_special_tokens: A python integer, number of special tokens. Default is 3: ['<unk>', '<s>', '</s>']
            name: A python string, embedding variable's name
            dtype: Data type
            scope: A python string, variable scope
        """
        super(PretrainedEmbedding, self).__init__(vocab_file, name, dtype, scope)

        self.pretrained_file = pretrained_file
        self.num_special_tokens = num_special_tokens

    def _embedding(self, inputs, length, params):
        vocab = self._load_vocab()
        # special tokens should put in the head of vocab
        special_tokens = vocab[:self.num_special_tokens]
        embedding_dict, embedding_size = self._load_pretrained_embedding()
        for token in special_tokens:
            if token not in embedding_dict:
                embedding_dict[token] = [0.0] * embedding_size

        embedding_matrix = np.array([embedding_dict[t] for t in vocab], dtype=self.dtype.as_numpy_dtype)
        embedding_matrix = tf.constant(embedding_matrix)

        # pretrained embedding
        embedding_const = tf.slice(input_=embedding_matrix, begin=[self.num_special_tokens, 0], size=[-1, -1])

        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE) as scope:
            # special tokens' embedding is trainable
            embedding_variable = tf.get_variable(
                name="embedding_variable",
                shape=[self.num_special_tokens, embedding_size],
                dtype=self.dtype)

            embedding = tf.concat([embedding_variable, embedding_const], axis=0)

            inputs_ids = self.str2id.lookup(inputs)
            embedded_inputs = tf.nn.embedding_lookup(embedding, inputs_ids)

        return embedded_inputs

    def _load_vocab(self):
        vocabs = []
        with codecs.getreader("utf8")(tf.gfile.GFile(self.vocab_file, "rb")) as f:
            for v in f:
                vocabs.append(v.strip('\n'))
        return vocabs

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
