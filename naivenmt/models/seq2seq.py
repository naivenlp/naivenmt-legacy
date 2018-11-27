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

from naivenmt.models.abstract_model import AbstractModel
from naivenmt.utils import dataset_utils
from naivenmt.utils import learning_rate_utils as lr_utils


class Seq2SeqModel(AbstractModel):

  def __init__(self,
               embedding,
               encoder,
               decoder,
               scope="seq2seq",
               dtype=tf.float32):
    self.embedding = embedding
    self.encoder = encoder
    self.decoder = decoder
    self.scope = scope
    self.dtype = dtype

  def input_fn(self, params, mode):
    return dataset_utils.build_dataset(params, mode)

  def model_fn(self, features, labels, mode, params, config=None):
    src = features['inputs']
    src_len = features['inputs_length']

    with tf.variable_scope(self.scope):
      # embedding source sequence
      src_inputs = self.embedding.encoder_embedding_input(src)

      # encode
      enc_outputs, enc_states = self.encoder.encode(mode, src_inputs, src_len)
      # TODO(luozhouyang) decide where to transpose the outputs of encoder
      if params.time_major:
        enc_outputs = tf.transpose(enc_outputs, perm=[1, 0, 2])

      # embedding target sequence
      labels_in = self.embedding.decoder_embedding_input(labels['tgt_in'])
      labels_len = labels['tgt_len']
      labels_out = self.embedding.decoder_embedding_input(labels['tgt_out'])
      new_labels = {
        "tgt_in": labels_in,
        "tgt_out": labels_out,
        "tgt_len": labels_len
      }

      # decode
      logits, predict_ids, dec_state = self.decoder.decode(
        mode, enc_outputs, enc_states, new_labels, src_len)

      if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = self.build_predictions(predict_ids, params)
        export_outputs = {
          "export_outputs": predictions
        }
        prediction_hooks = self.build_prediction_hooks()
        return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=predictions,
          prediction_hooks=prediction_hooks,
          export_outputs=export_outputs)

      # TODO(luozhouyang) sampled_softmax_loss
      loss = self.compute_loss(logits, new_labels, params)

      if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = self.build_train_op(loss, params)
        training_hooks = self.build_training_hooks()
        return tf.estimator.EstimatorSpec(
          mode=mode,
          train_op=train_op,
          training_hooks=training_hooks,
          loss=loss)

      if mode == tf.estimator.ModeKeys.EVAL:
        metric_ops = self.build_eval_metrics(
          predict_ids, labels, src_len, params)
        evaluation_hooks = self.build_evaluation_hooks()
        return tf.estimator.EstimatorSpec(
          mode=mode,
          eval_metric_ops=metric_ops,
          evaluation_hooks=evaluation_hooks,
          loss=loss)

  def build_predictions(self, predict_ids, params):
    raise NotImplementedError()

  def build_training_hooks(self):
    return []

  def build_evaluation_hooks(self):
    return []

  def build_prediction_hooks(self):
    return []

  def build_eval_metrics(self, predict_ids, labels, src_len, params):
    raise NotImplementedError()

  def compute_loss(self, logits, labels, params):
    target_output = labels['tgt_out']
    max_time_steps = tf.shape(target_output)[1]
    batch_size = tf.shape(target_output)[0]
    if params.time_major:
      target_output = tf.transpose(target_output, perm=[1, 0, 2])
      max_time_steps = tf.shape(target_output)[0]
      batch_size = tf.shape(target_output)[1]

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=target_output,
      logits=logits)
    target_weights = tf.sequence_mask(
      lengths=labels['tgt_len'],
      maxlen=max_time_steps,
      dtype=self.dtype)
    loss = tf.reduce_sum(cross_entropy * target_weights) / tf.to_float(
      batch_size)
    return loss

  def build_train_op(self, loss, params):
    if params.optimizer == "sgd":
      self.sgd_lr = tf.constant(params.learning_rate)
      self.sgd_lr = lr_utils.learning_rate_warmup(self.sgd_lr, params)
      self.sgd_lr = lr_utils.learning_rate_decay(self.sgd_lr, params)
      opt = tf.train.GradientDescentOptimizer(self.sgd_lr)
    elif params.optimizer == "adam":
      opt = tf.train.AdamOptimizer()
    else:
      raise ValueError("Unknown optimizer %s" % params.optimizer)
    gradients = tf.gradients(
      loss,
      tf.trainable_variables(),
      colocate_gradients_with_ops=params.colocate_gradients_with_ops)
    clipped_grads, grad_norm = tf.clip_by_global_norm(
      gradients, params.max_gradient_norm)
    train_op = opt.apply_gradients(
      zip(clipped_grads, grad_norm),
      tf.train.get_or_create_global_step())
    return train_op
