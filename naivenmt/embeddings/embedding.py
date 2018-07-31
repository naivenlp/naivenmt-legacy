import abc
import tensorflow as tf
import numpy as np
import codecs

VOCAB_SIZE_THRESHOLD = 50000


class EmbeddingInterface(abc.ABC):

  @abc.abstractmethod
  def encoder_embedding(self):
    """Create embedding for encoder."""
    raise NotImplementedError()

  @abc.abstractmethod
  def decoder_embedding(self):
    """Create embedding for decoder."""
    raise NotImplementedError()

  @abc.abstractmethod
  def encoder_embedding_input(self, inputs):
    """Create encoder embedding input.

    Args:
      inputs: ids of source

    Returns:
      embedding presentation of inputs
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def decoder_embedding_input(self, inputs):
    """Create decoder embedding input.

    Args:
      inputs: ids of target

    Returns:
      embedding presentation of inputs
    """
    raise NotImplementedError()


class Embedding(EmbeddingInterface):

  def __init__(self,
               src_vocab_size,
               tgt_vocab_size,
               share_vocab,
               src_embedding_size,
               tgt_embedding_size,
               num_partitions=0,
               dtype=tf.float32,
               src_vocab_file=None,
               tgt_vocab_file=None,
               src_embedding_file=None,
               tgt_embedding_file=None,
               scope="embedding"):
    self.share_vocab = share_vocab
    self.src_vocab_file = src_vocab_file
    self.tgt_vocab_file = tgt_vocab_file
    self.src_vocab_size = src_vocab_size
    self.tgt_vocab_size = tgt_vocab_size
    self.src_embedding_size = src_embedding_size
    self.tgt_embedding_size = tgt_embedding_size
    self.src_embedding_file = src_embedding_file
    self.tgt_embedding_file = tgt_embedding_file
    self.num_partitions = num_partitions
    self.dtype = dtype
    self.scope = scope
    self._encoder_embedding = None
    self._decoder_embedding = None

    self._embedding()

  def _embedding(self):
    if self.num_partitions <= 1:
      partitioner = None
    else:
      partitioner = tf.fixed_size_partitioner(num_shards=self.num_partitions)

    if (self.src_embedding_file or self.tgt_embedding_file) and partitioner:
      raise ValueError(
        "Can't set num_partitions > 1 when using pretrained embedding")

    with tf.variable_scope(self.scope, dtype=self.dtype,
                           partitioner=partitioner) as scope:
      if self.share_vocab:
        if self.src_vocab_size != self.tgt_vocab_size:
          raise ValueError(
            "Share embedding but different src/tgt vocab sizes %d vs. %d" % (
              self.src_vocab_size, self.tgt_vocab_size))
        assert self.src_embedding_size == self.tgt_embedding_size
        vocab_file = self.src_vocab_file or self.tgt_vocab_file
        embedding_file = self.src_embedding_file or self.tgt_embedding_file
        self._encoder_embedding = self._create_or_load_embeddings(
          "embedding_share",
          vocab_file,
          embedding_file,
          self.src_vocab_size,
          self.src_embedding_size,
          self.dtype)
        self._decoder_embedding = self.encoder_embedding
      else:
        with tf.variable_scope("encoder", partitioner=partitioner):
          self._encoder_embedding = self._create_or_load_embeddings(
            "encoder_embedding",
            self.src_vocab_file,
            self.src_embedding_file,
            self.src_vocab_size,
            self.src_embedding_size,
            self.dtype)
        with tf.variable_scope("decoder", partitioner=partitioner):
          self._decoder_embedding = self._create_or_load_embeddings(
            "decoder_embedding",
            self.tgt_vocab_file,
            self.tgt_embedding_file,
            self.tgt_vocab_size,
            self.tgt_embedding_size,
            self.dtype)
      return self.encoder_embedding, self._decoder_embedding

  def _create_or_load_embeddings(self,
                                 name,
                                 vocab_file,
                                 embedding_file,
                                 vocab_size,
                                 embedding_size,
                                 dtype):
    if vocab_file and embedding_file:
      embedding = self._create_pretrained_embedding(vocab_file, embedding_file)
    else:
      with tf.device(self._create_embedding_device(vocab_size)):
        embedding = tf.get_variable(
          name=name,
          shape=[vocab_size, embedding_size],
          dtype=dtype)
    return embedding

  def _create_pretrained_embedding(self,
                                   vocab_file,
                                   embedding_file,
                                   num_trainable_tokens=3,
                                   dtype=tf.float32,
                                   scope="pretrained_embedding"):
    vocab, _ = self._load_vocab(vocab_file)
    trainable_tokens = vocab_file[:num_trainable_tokens]
    embedding_dict, embedding_size = self._load_embedding_txt(embedding_file)
    for token in trainable_tokens:
      if token in embedding_dict:
        embedding_dict[token] = [0.0] * embedding_size
    embedding_matrix = np.array(
      [embedding_dict[token] for token in vocab], dtype=dtype.as_numpy_dtype)
    embedding_matrix = tf.constant(embedding_matrix)
    embedding_matrix_const = tf.slice(
      embedding_matrix, [num_trainable_tokens, 0], [-1, -1])
    with tf.variable_scope(scope, dtype=dtype):
      embedding_matrix_variable = tf.get_variable(
        "embedding_matrix_variable",
        [num_trainable_tokens, embedding_size])
    return tf.concat([embedding_matrix_variable, embedding_matrix_const], 0)

  @staticmethod
  def _load_vocab(vocab_file):
    vocab = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
      vocab_size = 0
      for word in f:
        vocab_size += 1
        vocab.append(word.strip())
    return vocab, vocab_size

  @staticmethod
  def _load_embedding_txt(embedding_file):
    embedding_dict = dict()
    embedding_size = None
    with codecs.getreader("utf-8")(tf.gfile.GFile(embedding_file, "rb")) as f:
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

  @staticmethod
  def _create_embedding_device(vocab_size):
    if vocab_size > VOCAB_SIZE_THRESHOLD:
      return "/cpu:0"
    else:
      return "/gpu:0"

  @property
  def encoder_embedding(self):
    return self._encoder_embedding

  @property
  def decoder_embedding(self):
    return self._decoder_embedding

  def encoder_embedding_input(self, inputs):
    return tf.nn.embedding_lookup(self.encoder_embedding, inputs)

  def decoder_embedding_input(self, inputs):
    return tf.nn.embedding_lookup(self.decoder_embedding, inputs)
