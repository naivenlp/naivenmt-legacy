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

from naivenmt.models.abstract_model import AbstractModel
from naivenmt.utils import dataset_utils


class Seq2SeqModel(AbstractModel):

  def __init__(self,
               params,
               encoder,
               decoder,
               scope="seq2seq"):
    self.encoder = encoder
    self.decoder = decoder

    self.params = params

    self.scope = scope

  def input_fn(self, params, mode):
    return dataset_utils.build_dataset(params, mode)

  def model_fn(self, features, labels, mode, params, config=None):
    src = features['inputs']
    src_len = features['inputs_length']

    src_str2idx = lookup_ops.index_table_from_file(
      params.source_vocab_file, default_value=0)
    src_ids = src_str2idx.lookup(src)

    tgt_str2idx = lookup_ops.index_table_from_file(
      params.target_vocab_file, default_value=0)

    training = mode == tf.estimator.ModeKeys.TRAIN

    with tf.variable_scope(self.scope):
      with tf.variable_scope("encoder"):
        enc_output, enc_state = self.encoder.encode(mode, src_ids)

      with tf.variable_scope("decoder"):
        dec_output, dec_state = self.decoder.decode(
          mode, enc_output, enc_state, labels, src_len)

      if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = self.build_predictions(dec_output, dec_state)
        export_outputs = {
          "export_outputs": predictions
        }
        prediction_hooks = self.build_prediction_hooks()
        return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=predictions,
          prediction_hooks=prediction_hooks,
          export_outputs=export_outputs)

      with tf.variable_scope("loss"):
        loss = self.compute_loss(dec_output, dec_state, labels)

      if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = self.build_train_op(params)
        training_hooks = self.build_training_hooks()
        return tf.estimator.EstimatorSpec(
          mode=mode,
          train_op=train_op,
          training_hooks=training_hooks,
          loss=loss)

      if mode == tf.estimator.ModeKeys.EVAL:
        metric_ops = self.build_eval_metrics()
        evaluation_hooks = self.build_evaluation_hooks()
        return tf.estimator.EstimatorSpec(
          mode=mode,
          eval_metric_ops=metric_ops,
          evaluation_hooks=evaluation_hooks,
          loss=loss)

  def build_predictions(self, dec_output, dec_state):
    raise NotImplementedError()
