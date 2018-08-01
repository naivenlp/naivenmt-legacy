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
    embedding = Embedding(src_vocab_size=configs.source_vocab_size,
                          tgt_vocab_size=configs.target_vocab_size,
                          share_vocab=params.share_vocab,
                          src_embedding_size=params.source_embedding_size,
                          tgt_embedding_size=params.target_embedding_size,
                          src_vocab_file=configs.source_vocab_file,
                          tgt_vocab_file=configs.target_vocab_file,
                          src_embedding_file=configs.source_embedding_file,
                          tgt_embedding_file=configs.target_embedding_file)
    encoder = GNMTEncoder(params=params, embedding=embedding)
    decoder = GNMTDecoder(configs=configs,
                          params=params,
                          embedding=embedding,
                          sos=configs.sos,
                          eos=configs.eos)
    super().__init__(inputter=inputter, encoder=encoder, decoder=decoder)
