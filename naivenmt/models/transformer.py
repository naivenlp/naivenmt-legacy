from naivenmt.decoders import TransformerDecoder
from naivenmt.embeddings import Embedding
from naivenmt.encoders import TransformerEncoder
from naivenmt.inputters import Inputter
from naivenmt.models import SequenceToSequence


# TODO(luozhouyang) Add hparams to config
class Transformer(SequenceToSequence):

  def __init__(self,
               params,
               scope=None,
               dtype=None,
               lifecycle_hooks=None,
               tensors_hooks=None):
    inputter = Inputter(params=params)
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
    encoder = TransformerEncoder(embedding,
                                 num_layers=6,
                                 model_dim=512,
                                 ffn_dim=2048,
                                 dropout=0.2,
                                 dtype=dtype)
    # TODO(luozhouyang) handle sos end eos
    decoder = TransformerDecoder(embedding,
                                 num_layers=6,
                                 model_dim=512,
                                 ffn_dim=2048,
                                 dropout=0.2,
                                 dtype=dtype)
    super(Transformer, self).__init__(inputter=inputter,
                                      encoder=encoder,
                                      decoder=decoder,
                                      scope=scope,
                                      dtype=dtype,
                                      lifecycle_hooks=lifecycle_hooks,
                                      tensors_hooks=tensors_hooks)

  def model_fn(self):
    pass

  def input_fn(self, mode):
    pass

  def serving_input_fn(self):
    pass
