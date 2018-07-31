from naivenmt.decoders import BasicDecoder
from naivenmt.embeddings import Embedding
from naivenmt.encoders import BasicEncoder
from naivenmt.inputters import Inputter
from naivenmt.models import SequenceToSequence


class BasicModel(SequenceToSequence):
  """Basic NMT model."""

  def __init__(self, configs, params, infer_file=None):
    inputter = Inputter(configs=configs,
                        params=params,
                        predict_file=infer_file)
    # TODO(luozhouyang) Add `source_vocab_size` and `target_vocab_size` to params
    # TODO(luozhouyang) Add `source_embedding_size` and `target_embedding_size` to params
    embedding = Embedding(src_vocab_size=params.source_vocab_size,
                          tgt_vocab_size=params.target_vocab_size,
                          share_vocab=params.share_vocab,
                          src_embedding_size=params.source_embedding_size,
                          tgt_embedding_size=params.target_embedding_size,
                          src_vocab_file=configs.src_vocab_file,
                          tgt_vocab_file=configs.tgt_vocab_file)
    encoder = BasicEncoder(params=params, embedding=embedding)
    decoder = BasicDecoder(params=params, embedding=embedding,
                           sos=configs.sos, eos=configs.eos)
    super().__init__(inputter=inputter, encoder=encoder, decoder=decoder)
