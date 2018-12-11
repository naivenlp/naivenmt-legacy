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

import numpy as np
import tensorflow as tf

from naivenmt.configs import HParamsBuilder
from naivenmt.decoders import BasicDecoder
from naivenmt.embeddings import Embedding

BATCH_SIZE = 8
TIME_STEPS = 5
DEPTH = 4


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


def get_default_test_configs():
  return {
    "num_encoder_layers": 2,
    "num_encoder_residual_layers": 1,
    "num_decoder_layers": 2,
    "num_decoder_residual_layers": 1,
    "beam_width": 0,
    "length_penalty_weight": 0,
    "encoder_type": "bi",
    "unit_type": "gru",
    "num_units": 4,
    "forget_bias": 1.0,
    "dropout": 0.5,
    "time_major": True,
    "embed_prefix": get_testdata_file("test_embed"),
    "vocab_prefix": get_testdata_file("test_embed_vocab"),
    "source_embedding_size": 4,
    "target_embedding_size": 4,
    "share_vocab": False,
    "infer_batch_size": 1,
    "tgt_max_len_infer": 40,
    "sampling_temperature": 0.0,
    "random_seed": None
  }


def get_test_configs(update_configs):
  default_configs = get_default_test_configs()
  default_configs.update(update_configs)
  return default_configs


def get_params(update_configs):
  configs = get_test_configs(update_configs)
  hparams = HParamsBuilder(configs).build()
  return hparams


def get_encoder_test_inputs():
  inputs = np.array([
    [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]],
    [[2, 3, 1, 4], [3, 4, 5, 1], [6, 8, 2, 5], [2, 4, 6, 7], [0, 0, 0, 0]],
    [[3, 5, 6, 7], [1, 2, 8, 5], [4, 5, 6, 3], [2, 3, 4, 5], [1, 2, 6, 7]],
    [[2, 3, 9, 5], [5, 7, 2, 1], [6, 2, 3, 8], [0, 0, 0, 0], [0, 0, 0, 0]],
    [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]],
    [[2, 3, 1, 4], [3, 4, 5, 1], [6, 8, 2, 5], [2, 4, 6, 7], [0, 0, 0, 0]],
    [[3, 5, 6, 7], [1, 2, 8, 5], [4, 5, 6, 3], [2, 3, 4, 5], [1, 2, 6, 7]],
    [[2, 3, 9, 5], [5, 7, 2, 1], [6, 2, 3, 8], [0, 0, 0, 0], [0, 0, 0, 0]],
  ], dtype=tf.float32.as_numpy_dtype)
  inputs_length = np.array([5, 4, 5, 3, 5, 4, 5, 3],
                           dtype=tf.int32.as_numpy_dtype)
  return inputs, inputs_length


def get_embedding(hparams):
  return Embedding(src_vocab_size=hparams.source_vocab_size,
                   tgt_vocab_size=hparams.target_vocab_size,
                   src_vocab_file=hparams.source_vocab_file,
                   tgt_vocab_file=hparams.target_vocab_file,
                   share_vocab=hparams.share_vocab,
                   src_embedding_size=hparams.source_embedding_size,
                   tgt_embedding_size=hparams.target_embedding_size)


def build_basic_decoder(configs):
  hparams = get_params(configs)
  embeddings = get_embedding(hparams)
  sos_id = tf.to_int32(1)
  eos_id = tf.to_int32(2)
  decoder = BasicDecoder(
    params=hparams,
    embedding=embeddings,
    sos_id=sos_id,
    eos_id=eos_id)
  return decoder


def get_encoder_outputs():
  outputs = np.array([
    [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]],
    [[2, 3, 1, 4], [3, 4, 5, 1], [6, 8, 2, 5], [2, 4, 6, 7], [0, 0, 0, 0]],
    [[3, 5, 6, 7], [1, 2, 8, 5], [4, 5, 6, 3], [2, 3, 4, 5], [1, 2, 6, 7]],
    [[2, 3, 9, 5], [5, 7, 2, 1], [6, 2, 3, 8], [0, 0, 0, 0], [0, 0, 0, 0]],
    [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]],
    [[2, 3, 1, 4], [3, 4, 5, 1], [6, 8, 2, 5], [2, 4, 6, 7], [0, 0, 0, 0]],
    [[3, 5, 6, 7], [1, 2, 8, 5], [4, 5, 6, 3], [2, 3, 4, 5], [1, 2, 6, 7]],
    [[2, 3, 9, 5], [5, 7, 2, 1], [6, 2, 3, 8], [0, 0, 0, 0], [0, 0, 0, 0]],
  ], dtype=tf.float32.as_numpy_dtype)
  outputs = tf.convert_to_tensor(outputs)
  outputs = tf.transpose(outputs, perm=[1, 0, 2])

  outputs_length = np.array([5, 4, 5, 3, 5, 4, 5, 3],
                            dtype=tf.int32.as_numpy_dtype)
  outputs_length = tf.convert_to_tensor(outputs_length)
  return outputs, outputs_length


def get_labels():
  tgt_in = tf.placeholder_with_default(
    np.random.randn(TIME_STEPS, BATCH_SIZE, DEPTH).astype(
      tf.float32.as_numpy_dtype()),
    shape=(None, None, DEPTH))
  tgt_out = tf.placeholder_with_default(
    np.random.randn(TIME_STEPS, BATCH_SIZE, DEPTH).astype(
      tf.float32.as_numpy_dtype()),
    shape=(None, None, DEPTH))
  lens = tf.constant(
    np.random.randn(BATCH_SIZE).astype(tf.int32.as_numpy_dtype(0)))
  tgt_len = tf.convert_to_tensor(lens, dtype=tf.int32)
  labels = {
    "tgt_in": tgt_in,
    "tgt_out": tgt_out,
    "tgt_len": tgt_len
  }
  return labels


def get_random_states(num_layers):
  states = [
    tf.placeholder_with_default(
      np.random.randn(BATCH_SIZE, DEPTH).astype(tf.float32.as_numpy_dtype()),
      shape=(BATCH_SIZE, DEPTH))
    for _ in range(num_layers)
  ]
  return states


def get_uni_lstm_encoder_results(num_layers):
  outputs, outputs_length = get_encoder_outputs()
  states_c = get_random_states(num_layers)
  states_h = get_random_states(num_layers)
  states = []
  for c, h in zip(states_c, states_h):
    states.append((c, h))
  states = tuple(states)
  return outputs, outputs_length, states


def get_uni_layer_norm_lstm_encoder_results(num_layers):
  return get_uni_lstm_encoder_results(num_layers)


def get_uni_nas_encoder_results(num_layers):
  return get_uni_lstm_encoder_results(num_layers)


def get_uni_gru_encoder_results(num_layers):
  outputs, outputs_length = get_encoder_outputs()
  states = get_random_states(num_layers)
  states = tuple(states)
  return outputs, outputs_length, states
