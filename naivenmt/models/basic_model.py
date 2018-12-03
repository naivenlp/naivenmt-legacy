# Copyright 2018 luozhouyang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
from tensorflow.python.ops import lookup_ops

from naivenmt.decoders.basic_decoder import BasicDecoder
from naivenmt.embeddings.embedding import Embedding
from naivenmt.encoders.basic_encoder import BasicEncoder
from naivenmt.models import Seq2SeqModel


class BasicModel(Seq2SeqModel):
  """Basic NMT model."""

  def __init__(self,
               params,
               scope="basic_model",
               dtype=tf.float32):
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
    encoder = BasicEncoder(params=params,
                           scope="basic_encoder",
                           dtype=dtype)
    tgt_str2idx = lookup_ops.index_table_from_file(params.target_vocab_file,
                                                   default_value=0)
    sos_id = tgt_str2idx.lookup(params.sos)
    eos_id = tgt_str2idx.lookup(params.eos)
    decoder = BasicDecoder(params=params,
                           embedding=embedding,
                           sos_id=sos_id,
                           eos_id=eos_id,
                           dtype=dtype)
    super(BasicModel, self).__init__(
      embedding=embedding,
      encoder=encoder,
      decoder=decoder,
      scope=scope,
      dtype=dtype)
