import abc

import tensorflow as tf

from naivenmt import utils


class AbstractModel(abc.ABC):

  def __init__(self, scope=None, inputter=None, dtype=None):
    self.scope = scope
    self.inputter = inputter
    if not dtype and inputter:
      self.dtype = inputter.dtype
    else:
      self.dtype = dtype or tf.float32

  @abc.abstractmethod
  def model_fn(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def input_fn(self, mode, params):
    raise NotImplementedError()

  @abc.abstractmethod
  def serving_input_fn(self):
    raise NotImplementedError()


class SequenceToSequence(AbstractModel):

  def __init__(self,
               inputter=None,
               encoder=None,
               decoder=None,
               scope="seq2seq",
               dtype=None):
    super().__init__(scope, inputter, dtype)
    self.inputter = inputter
    self.encoder = encoder
    self.decoder = decoder

  def _build(self, features, labels, params, mode, configs):
    with tf.variable_scope(self.scope, dtype=self.dtype):
      encoder_outputs, encoder_state = self._encode(
        mode, features, params, configs)
      logits, sample_id, final_context_state = self._decode(
        mode, encoder_outputs, encoder_state, labels, params, configs)
      if mode != tf.estimator.ModeKeys.PREDICT:
        with tf.device(utils.get_device_str(params.num_encoder_layers - 1,
                                            params.num_gpus)):
          loss = self._compute_loss(logits, params)
      else:
        loss = None
      return logits, loss, final_context_state, sample_id

  def model_fn(self):
    def _model_fn(features, labels, params, mode, config):
      logits, loss, _, sample_id = self._build(
        features, labels, params, mode, config)
      predictions = None
      if mode != tf.estimator.ModeKeys.TRAIN:
        predictions = self._decode_predictions(sample_id, params.time_major)
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
        export_outputs[k] = tf.estimator.export.PredictOutput(predictions)
        return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=predictions,
          export_outputs=export_outputs)
      else:
        raise ValueError("Invalid mode: %s" % mode)

    return _model_fn

  def input_fn(self, mode, params):
    iterator = self.inputter.iterator(mode, params)
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    return lambda: iterator.get_next()

  def serving_input_fn(self):
    return lambda: self._serving_input_fn()

  def _serving_input_fn(self):
    return self.inputter.serving_input_receiver()

  def _encode(self, mode, features, params, configs):
    # embedding_input = self.inputter.embedding_input(features)
    sequence_length = self.inputter.source_sequence_length()
    return self.encoder.encode(
      mode, features, sequence_length, params, configs)

  def _decode(self,
              mode, encoder_outputs, encoder_state, labels, params, configs):
    # embedding_input = self.inputter.embedding_input(labels)
    src_seq_len = self.inputter.source_sequence_length()
    tgt_seq_len = self.inputter.target_sequence_length()
    return self.decoder.decode(
      mode,
      encoder_outputs,
      encoder_state,
      labels,
      src_seq_len,
      tgt_seq_len,
      params,
      configs)

  def _compute_loss(self, logits, params):
    target_output = self.inputter.target_output
    if params.time_major:
      target_output = tf.transpose(target_output)
    time_axis = 0 if params.time_major else 1
    max_time = target_output.shape[time_axis] or tf.shape(target_output)[
      time_axis]
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=target_output, logits=logits)
    target_weights = tf.sequence_mask(
      self.inputter.target_sequence_length, max_time, dtype=logits.dtype)
    if params.time_major:
      target_weights = tf.transpose(target_weights)
    loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(
      params.batch_size)
    return loss

  def _train_op(self, loss, params):
    global_steps = tf.Variable(0, trainable=False)
    lr = tf.constant(params.learning_rate)
    lr = self._warmup_lr(lr, global_steps, params)
    lr = self._decay_lr(lr, global_steps, params)
    opt = self._optimizer(lr, params)
    gradients = tf.gradients(
      loss,
      params,
      colocate_gradients_with_ops=params.colocate_gradients_with_ops)
    clipped_grads, grad_norm_summary, grad_norm = self._clip_gradients(
      gradients, params.max_gradient_norm)
    update = opt.apply_gradients(
      zip(clipped_grads, params), global_step=global_steps)
    tf.summary.merge([tf.summary.scalar("lr", lr),
                      tf.summary.scalar("train_loss", loss)
                      ] + grad_norm_summary)
    return update

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
    gradient_norm_summary = [
      tf.summary.scalar("grad_norm", gradient_norm),
      tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients))]
    return clipped_gradients, gradient_norm_summary, gradient_norm

  def _eval_metric_ops(self, features, labels, predictions):
    return None

  def _decode_predictions(self, sample_id, time_major):
    sample_words = self.inputter.reverse_target_vocab_table.lookup(
      tf.to_int64(sample_id))
    if time_major:
      sample_words = sample_words.transpose()
    elif sample_words.ndim == 3:
      sample_words = sample_words.transpose([2, 0, 1])
    return sample_words
