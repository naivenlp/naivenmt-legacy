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

import os

import tensorflow as tf

from naivenmt.configs import HParamsBuilder
from naivenmt.embeddings import Embedding
from naivenmt.encoders import BasicEncoder


def get_testdata_dir():
  return os.path.abspath(os.path.join(os.pardir, "../", "testdata"))


def get_testdata_file(filename):
  return os.path.join(get_testdata_dir(), filename)


def get_module_dir(dirname):
  """Get dir path of module under naivenmt."""
  return os.path.abspath(os.path.join(os.pardir, dirname))


def get_file_path(module, file):
  """Get file's abs path."""
  naivenmt_dir = os.path.pardir
  return os.path.abspath(os.path.join(naivenmt_dir, module, file))


def get_configs():
  return {
    "num_encoder_layers": 2,
    "num_encoder_residual_layers": 1,
    "encoder_type": "bi",
    "unit_type": "gru",
    "num_units": 256,
    "forget_bias": 1.0,
    "dropout": 0.5,
    "time_major": True,
    "embed_prefix": get_testdata_file("test_embed"),
    "vocab_prefix": get_testdata_file("test_embed_vocab"),
    "source_embedding_size": 4,
    "target_embedding_size": 4,
    "share_vocab": False
  }


def get_embedding(hparams):
  return Embedding(src_vocab_size=hparams.source_vocab_size,
                   tgt_vocab_size=hparams.target_vocab_size,
                   src_vocab_file=hparams.source_vocab_file,
                   tgt_vocab_file=hparams.target_vocab_file,
                   share_vocab=hparams.share_vocab,
                   src_embedding_size=hparams.source_embedding_size,
                   tgt_embedding_size=hparams.target_embedding_size)


def get_basic_encode_results(update_configs):
  configs = get_configs()
  configs.update(update_configs)
  hparams = HParamsBuilder(configs).build()
  embedding = get_embedding(hparams)
  encoder = BasicEncoder(params=hparams)

  inputs = embedding.encoder_embedding_input(
    tf.constant([['Hello', 'World'], ['Hello', 'World'], ['Hello', '<PAD>']],
                dtype=tf.string))

  outputs, states = encoder.encode(
    mode=tf.estimator.ModeKeys.TRAIN,
    sequence_inputs=inputs,
    sequence_length=tf.constant([2, 2, 1], dtype=tf.int32))
  return outputs, states, hparams
