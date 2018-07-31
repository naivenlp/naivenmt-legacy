import tensorflow as tf

from naivenmt.decoders import BasicDecoder
from naivenmt.embeddings.embedding import Embedding
from naivenmt.encoders import BasicEncoder
from naivenmt.inputters import Inputter
from naivenmt.models import SequenceToSequence


class BasicModel(SequenceToSequence):

  def __init__(self, configs):
    inputter = Inputter(config=configs, dtype=tf.float32)
    embedding = Embedding(src_vocab_size=100000,
                          tgt_vocab_size=100000,
                          share_vocab=False,
                          src_embedding_size=256,
                          tgt_embedding_size=256,
                          src_vocab_file=configs.src_vocab_file,
                          tgt_vocab_file=configs.tgt_vocab_file)
    encoder = BasicEncoder(embedding=embedding)
    decoder = BasicDecoder(embedding=embedding)
    super().__init__(inputter, encoder, decoder)
