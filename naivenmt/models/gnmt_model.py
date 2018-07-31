from naivenmt.decoders import GNMTDecoder
from naivenmt.embeddings import Embedding
from naivenmt.encoders import GNMTEncoder
from naivenmt.inputters import Inputter
from naivenmt.models import SequenceToSequence


class GNMTModel(SequenceToSequence):
  """GNMT model."""

  def __init__(self, configs, params, predict_file=None):
    inputter = Inputter(configs=configs,
                        params=params,
                        predict_file=predict_file)
    embedding = Embedding(src_vocab_size=params.source_vocab_size,
                          tgt_vocab_size=params.target_vocab_size,
                          share_vocab=params.share_vocab,
                          src_embedding_size=params.source_embedding_size,
                          tgt_embedding_size=params.target_embedding_size,
                          src_vocab_file=configs.src_vocab_file,
                          tgt_vocab_file=configs.tgt_vocab_file)
    encoder = GNMTEncoder(params=params, embedding=embedding)
    decoder = GNMTDecoder(params=params, embedding=embedding,
                          sos=configs.sos, eos=configs.eos)
    super().__init__(inputter, encoder, decoder)
