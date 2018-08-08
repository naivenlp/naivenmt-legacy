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

import abc

import tensorflow as tf

from naivenmt.inputters.features import Features
from naivenmt.inputters.labels import Labels
from naivenmt.utils import add_dict_to_collection, add_to_collection


class ModelInterface(abc.ABC):
  """NMT models' interface, for integrating with Estimator API."""

  def __init__(self, inputter=None, scope=None, dtype=None):
    """Init model.

    Args:
      inputter: input data for training, evaluation and inference
      scope: variable scope
      dtype: dtype, default value tf.float32
    """
    self.scope = scope or "seq2seq"
    self.inputter = inputter
    self.dtype = dtype or tf.float32

  @abc.abstractmethod
  def model_fn(self):
    """Create a model_fn for estimator,
        so it's signature and return are the same as estimator's mode_fn.

    Returns:
      model_fn: estimator API's model_fn
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def input_fn(self, mode):
    """Create a input_fn for estimator api,
        so it's signature and return are the same as estimator's input_fn.

    Args:
        mode: mode, one of tf.estimator.ModeKeys

    Returns:
        input_fn: estimator API's input fn
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def serving_input_fn(self):
    """Create a serving_input_fn to export model and serve it on tensorflow serving."""
    raise NotImplementedError()


class SequenceToSequence(ModelInterface):
  """Sequence to sequence base model."""

  def __init__(self,
               inputter=None,
               encoder=None,
               decoder=None,
               scope=None,
               dtype=None,
               lifecycle_hooks=None,
               tensors_hooks=None):
    """Init seq2seq model.

    Args:
      inputter: input data for training, evaluation and inference
      encoder: encode source inputs
      decoder: decode target inputs
      scope: variable scope
      dtype: dtype of variables
      lifecycle_hooks: model's lifecycle listeners
      tensors_hooks: tensors' listeners
    """
    super().__init__(inputter=inputter, scope=scope, dtype=dtype)
    self.inputter = inputter
    self.encoder = encoder
    self.decoder = decoder
    self.lifecycle_hooks = lifecycle_hooks
    self.tensors_hooks = tensors_hooks

  def _build(self, features, labels, params, mode, configs):
    """Encode source inputs and decode target inputs, calculate loss.

    Args:
      features: first returned value of input_fn, instance of `naivenmt.inputters.Features`
      labels: second returned value of input_fn, instance of `naivenmt.inputters.Labels`
      params: hparams
      mode: mode
      configs: configs

    Returns:
      logits: logits
      loss: loss of train and eval, None for predict
      final_context_state: final context state of decoder
      sample_id: tokens ids from decoder
    """
    with tf.variable_scope(self.scope, dtype=self.dtype,
                           initializer=self._initializer(mode, params)):

      self._lifecycle_before_encode(mode, features)
      encoder_outputs, encoder_state = self._encode(mode, features)
      self._lifecycle_after_encode(
        mode, features, encoder_outputs, encoder_state)
      src_seq_len = features.source_sequence_length
      self._lifecycle_before_decode(
        mode, encoder_outputs, encoder_state, labels, src_seq_len)
      logits, sample_id, final_context_state = self._decode(
        mode, encoder_outputs, encoder_state, labels, src_seq_len)
      if mode != tf.estimator.ModeKeys.PREDICT:
        with tf.device(self._get_device_str(params.num_encoder_layers - 1,
                                            params.num_gpus)):
          loss = self._compute_loss(logits, params, labels)
      else:
        loss = None
      self._lifecycle_after_decode(
        mode, logits, loss, sample_id, final_context_state)
      return logits, loss, final_context_state, sample_id

  def model_fn(self):
    """Create model fn for estimator."""

    def _model_fn(features, labels, params, mode, config):
      logits, loss, _, sample_id = self._build(
        features, labels, params, mode, config)
      predictions = None
      if mode != tf.estimator.ModeKeys.TRAIN:
        predictions = self._decode_predictions(sample_id, params.time_major)
        # Add predictions tensors to tf collection for hooking.
        add_dict_to_collection("predictions", predictions)
      if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = self._train_op(loss, params)
        return tf.estimator.EstimatorSpec(
          mode=mode,
          loss=loss,
          train_op=train_op)
      elif mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = self._eval_metric_ops(features, labels, predictions)
        return tf.estimator.EstimatorSpec(
          mode=mode,
          loss=loss,
          eval_metric_ops=eval_metric_ops)
      elif mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {}
        k = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        # predictions is a tensor or a dict of tensors.
        # gRPC clients can get the result from serving and then parse
        # the tensors result to text.
        export_outputs[k] = tf.estimator.export.PredictOutput(predictions)
        return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=predictions,
          export_outputs=export_outputs)
      else:
        raise ValueError("Invalid mode: %s" % mode)

    return _model_fn

  def input_fn(self, mode):
    """Input fn.

    Args:
      mode: mode, to decide which iterator to generate features and labels

    Returns:
      features: features, instance of ``naivenmt.inputters.Features``
      labels: labels, instance of ``naivenmt.inputters.Labels``
    """

    def _input_fn():
      iterator = self.inputter.iterator(mode)
      tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
                           iterator.initializer)
      if mode != tf.estimator.ModeKeys.PREDICT:
        src_ids, tgt_input_ids, tgt_output_ids, src_len, tgt_len = (
          iterator.get_next())
        features = Features(source_ids=src_ids,
                            source_sequence_length=src_len)
        labels = Labels(target_input_ids=tgt_input_ids,
                        target_output_ids=tgt_output_ids,
                        target_sequence_length=tgt_len)
        return features, labels
      else:
        src_ids, src_len = iterator.get_next()
        features = Features(source_ids=src_ids,
                            source_sequence_length=src_len)
        return features

    return _input_fn

  def serving_input_fn(self):
    return lambda: self._serving_input_fn()

  def _serving_input_fn(self):
    receiver_tensors, features = self.inputter.serving_input()
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

  def _encode(self, mode, features):
    """Encode source inputs.

    Args:
      mode: mode
      features: input features, instance of `naivenmt.inputters.Features`

    Returns:
      encoder_outputs: encoder's outputs
      encoder_state: encoder's state
    """
    return self.encoder.encode(mode, features)

  def _decode(self, mode, encoder_outputs, encoder_state, labels, src_seq_len):
    """Decode target inputs.

    Args:
      mode: mode
      encoder_outputs: encoder's outputs
      encoder_state: encoder's state
      labels: input labels, instance of `naivenmt.inputters.Labels`
      src_seq_len: source sequence length

    Returns:
      logits: logits
      sample_id: sample id
      final_context_state: decoder's state
    """
    return self.decoder.decode(mode, encoder_outputs, encoder_state,
                               labels, src_seq_len)

  def _lifecycle_before_encode(self, mode, features):
    if self.lifecycle_hooks:
      for hook in self.lifecycle_hooks:
        hook.before_encode(mode, features)

  def _lifecycle_after_encode(self, mode, features, outputs, state):
    if self.lifecycle_hooks:
      for hook in self.lifecycle_hooks:
        hook.after_encode(mode, features, outputs, state)

  def _lifecycle_before_decode(self, mode, outputs, state, labels, src_seq_len):
    if self.lifecycle_hooks:
      for hook in self.lifecycle_hooks:
        hook.before_decode(mode, outputs, state, labels, src_seq_len)

  def _lifecycle_after_decode(self, mode, logits, loss, sample_id,
                              final_context_state):
    if self.lifecycle_hooks:
      for hook in self.lifecycle_hooks:
        hook.after_decode(mode, logits, loss, sample_id, final_context_state)

  @staticmethod
  def _compute_loss(logits, params, labels):
    target_output = labels.target_output_ids
    target_sequence_length = labels.target_sequence_length
    if params.time_major:
      target_output = tf.transpose(target_output)
    time_axis = 0 if params.time_major else 1
    max_time = target_output.shape[time_axis] or tf.shape(target_output)[
      time_axis]
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=target_output, logits=logits)
    target_weights = tf.sequence_mask(
      target_sequence_length, max_time, dtype=logits.dtype)
    if params.time_major:
      target_weights = tf.transpose(target_weights)
    loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(
      params.batch_size)
    return loss

  def _train_op(self, loss, params):
    global_steps = tf.train.get_or_create_global_step()
    lr = tf.constant(params.learning_rate)
    lr = self._warmup_lr(lr, global_steps, params)
    lr = self._decay_lr(lr, global_steps, params)
    opt = self._optimizer(lr, params)
    trainable_params = tf.trainable_variables()
    gradients = tf.gradients(
      loss,
      trainable_params,
      colocate_gradients_with_ops=params.colocate_gradients_with_ops)
    clipped_grads, grad_norm = self._clip_gradients(
      gradients, params.max_gradient_norm)
    update = opt.apply_gradients(
      zip(clipped_grads, trainable_params), global_step=global_steps)
    # tf.summary.merge([tf.summary.scalar("lr", lr),
    #                   tf.summary.scalar("train_loss", loss)
    #                   ] + grad_norm_summary)
    self._listen_tensors(global_steps, lr, gradients, clipped_grads, grad_norm)
    return update

  def _listen_tensors(self,
                      global_steps,
                      learning_rate,
                      gradients,
                      clipped_grads,
                      grad_norm):
    """Listen tensors creation and add tensors to tf collections for hooking.

    Args:
      global_steps: global steps
      learning_rate: learning rate
      gradients: gradients
      clipped_grads: clipped gradients
      grad_norm: grad_norm
    """
    if self.tensors_hooks:
      for hook in self.tensors_hooks:
        hook.on_global_steps_created(global_steps)
        hook.on_learning_rate_created(learning_rate)
        hook.on_gradients_created(gradients)
        hook.on_clipped_grads_created(clipped_grads)
        hook.on_grad_norm_created(grad_norm)
        hook.end()

    # Do not need to add global_steps, cause it has been added to collection
    add_to_collection("learning_rate", learning_rate)
    add_to_collection("gradients", gradients)
    add_to_collection("clipped_grads", clipped_grads)
    add_to_collection("grad_norm", grad_norm)

  def _warmup_lr(self, lr, global_steps, params):
    warmup_steps = params.warmup_steps
    warmup_scheme = params.warmup_scheme
    if warmup_scheme == "t2t":
      warmup_factor = tf.exp(tf.log(0.01) / warmup_steps)
      inverse_decay = warmup_factor ** (
        tf.to_float(warmup_steps - global_steps))
    else:
      raise ValueError("Unsupported warmup scheme: %s" % warmup_scheme)
    return tf.cond(
      global_steps < params.warmup_steps,
      lambda: inverse_decay * lr,
      lambda: lr,
      name="lr_rate_warmup_cond")

  def _decay_lr(self, lr, global_steps, params):
    start_decay_step = params.num_train_steps
    if params.decay_scheme in ["luong5", "luong10", "luong234"]:
      decay_factor = 0.5
      decay_times = 1
      if params.decay_scheme == "luong5":
        start_decay_step = int(params.num_train_steps / 2)
        decay_times = 5
      elif params.decay_scheme == "luong10":
        start_decay_step = int(params.num_train_steps / 2)
        decay_times = 10
      elif params.decay_scheme == "luong234":
        start_decay_step = int(params.num_train_steps * 2 / 3)
        decay_times = 4
      remain_steps = params.num_train_steps - start_decay_step
      decay_steps = int(remain_steps / decay_times)
    elif not params.decay_scheme:
      decay_steps = 0
      decay_factor = 1.0
    elif params.decay_scheme:
      raise ValueError("Invalid decay scheme: %s" % params.decay_scheme)
    return tf.cond(
      global_steps < start_decay_step,
      lambda: lr,
      lambda: tf.train.exponential_decay(
        lr,
        global_steps - start_decay_step,
        decay_steps,
        decay_factor,
        staircase=True),
      name="lr_rate_decay_cond")

  def _optimizer(self, lr, params):
    if params.optimizer == "sgd":
      opt = tf.train.GradientDescentOptimizer(lr)
      tf.summary.scalar("lr", lr)
    elif params.optimizer == "adam":
      opt = tf.train.AdamOptimizer(lr)
    else:
      raise ValueError("Unsupported optimizer: %s" % params.optimizer)
    return opt

  def _clip_gradients(self, gradients, max_gradient_norm):
    clipped_gradients, gradient_norm = (
      tf.clip_by_global_norm(gradients, max_gradient_norm))
    # gradient_norm_summary = [
    #   tf.summary.scalar("grad_norm", gradient_norm),
    #   tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients))]
    return clipped_gradients, gradient_norm

  # TODO(luozhouyang) add metric ops
  def _eval_metric_ops(self, features, labels, predictions):
    return None

  def _decode_predictions(self, sample_id, time_major):
    """Convert ids to strings."""
    sample_words = self.inputter.target_reverse_vocab_table.lookup(
      tf.to_int64(sample_id))
    if time_major:
      sample_words = sample_words.transpose()
    elif sample_words.ndim == 3:
      sample_words = sample_words.transpose([2, 0, 1])

    prediction_tensors_dict = {
      # "ids": sample_id,
      "words": sample_words
    }
    return prediction_tensors_dict

  def _initializer(self, mode, params):
    """Variables initializer. Only train mode need a initializer.

    Args:
      mode: mode
      params: hparams

    Returns:
      initializer: variables initializer if mode is train, otherwise None
    """
    if mode != tf.estimator.ModeKeys.TRAIN:
      return None
    init_op = params.init_op
    seed = params.random_seed
    init_weight = params.init_weight
    if init_op == "uniform":
      assert init_weight
      return tf.random_uniform_initializer(
        -init_weight, init_weight, seed=seed, dtype=self.dtype)
    elif init_op == "glorot_normal":
      return tf.keras.initializers.glorot_normal(seed=seed)
    elif init_op == "glorot_uniform":
      return tf.keras.initializers.glorot_uniform(seed=seed)
    else:
      raise ValueError("Unknown init_op: %s" % init_op)

  @staticmethod
  def _get_device_str(device_id, num_gpus):
    if num_gpus == 0:
      return "/cpu:0"
    device_str = "/gpu:%d" % (device_id % num_gpus)
    return device_str
