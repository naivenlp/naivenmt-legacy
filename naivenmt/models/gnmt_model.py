from naivenmt.decoders import GNMTDecoder
from naivenmt.embeddings import Embedding
from naivenmt.encoders import GNMTEncoder
from naivenmt.inputters import Inputter
from naivenmt.models import SequenceToSequence


class GNMTModel(SequenceToSequence):
  """GNMT model."""

  def __init__(self,
               params,
               predict_file=None,
               scope=None,
               dtype=None,
               lifecycle_hooks=None,
               tensors_hooks=None):
    inputter = Inputter(params=params,
                        predict_file=predict_file)
    embedding = Embedding(src_vocab_size=params.source_vocab_size,
                          tgt_vocab_size=params.target_vocab_size,
                          share_vocab=params.share_vocab,
                          src_embedding_size=params.source_embedding_size,
                          tgt_embedding_size=params.target_embedding_size,
                          src_vocab_file=params.source_vocab_file,
                          tgt_vocab_file=params.target_vocab_file,
                          src_embedding_file=params.source_embedding_file,
                          tgt_embedding_file=params.target_embedding_file,
                          dtype=dtype)
    encoder = GNMTEncoder(params=params,
                          embedding=embedding,
                          dtype=dtype)
    decoder = GNMTDecoder(params=params,
                          embedding=embedding,
                          sos=params.sos,
                          eos=params.eos,
                          dtype=dtype)
    super().__init__(inputter=inputter,
                     encoder=encoder,
                     decoder=decoder,
                     scope=scope,
                     dtype=dtype,
                     lifecycle_hooks=lifecycle_hooks,
                     tensors_hooks=tensors_hooks)
