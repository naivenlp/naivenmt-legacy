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

from naivenmt.decoders.attention_decoder import AttentionDecoder
from naivenmt.embeddings.embedding import Embedding
from naivenmt.encoders.basic_encoder import BasicEncoder
from naivenmt.inputters.inputter import Inputter
from naivenmt.models.sequence_to_sequence import SequenceToSequence


class AttentionModel(SequenceToSequence):
  """Attention NMT model."""

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
    encoder = BasicEncoder(params=params,
                           embedding=embedding,
                           dtype=dtype)
    decoder = AttentionDecoder(params=params,
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
