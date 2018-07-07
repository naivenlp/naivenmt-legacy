import abc

import tensorflow as tf

from . import utils


class BaseModel(abc.ABC):

  def __init__(self,
               name,
               features_inputter,
               labels_inputter,
               estimator,
               dtype=None):
    self.name = name
    self.features_inputter = features_inputter
    self.labels_inputter = labels_inputter
    self.estimator = estimator
    self.dtype = dtype if dtype else self.features_inputter.dtype

  def __call__(self, features, labels, params, mode, config):
    return self._build(features, labels, params, mode, config)

  def model_fn(self):
    def _loss_op(features, labels, params, mode, config):
      logits, _ = self._build(features, labels, params, mode, config)
      return self._compute_loss(features, labels, logits, params, mode)

    def _extract_loss(loss):
      actual_loss = loss
      tboard_loss = loss
      tf.summary.scalar("loss", tboard_loss)
      return actual_loss

    def _model_fn(features, labels, params, mode, config):
      if mode == tf.estimator.ModeKeys.TRAIN:
        with tf.variable_scope(self.name, initializer=self._initializer(params)):
          pass
        loss = _loss_op(features, labels, params, mode, config)
        loss = _extract_loss(loss)
        train_op = self._optimize(loss, params)
        return tf.estimator.EstimatorSpec(
          mode=mode,
          loss=loss,
          train_op=train_op)
      elif mode == tf.estimator.ModeKeys.EVAL:
        with tf.variable_scope(self.name):
          logits, predictions = self._build(features, labels, params, mode, config)
          loss = self._compute_loss(features, labels, logits, params, mode)
        loss = _extract_loss(loss)
        eval_metric_ops = self._compute_metrics(features, labels,
                                                predictions)
        if predictions:
          utils.add_dict_to_collections("predictions", predictions)
        return tf.estimator.EstimatorSpec(
          mode=mode,
          loss=loss,
          eval_metric_ops=eval_metric_ops)
      elif mode == tf.estimator.ModeKeys.PREDICT:
        with tf.variable_scope(self.name):
          _, predictions = self._build(features, labels, params, mode, config)
        export_outputs = {
          tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
            predictions)}
        return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=predictions,
          export_outputs=export_outputs)
      else:
        raise ValueError("Invalid mode %s" % mode)

    return _model_fn

  def input_fn(self, mode):
    return lambda: self._input_fn(mode)

  def _input_fn(self, mode):
    features_dataset = self.features_inputter.build_dataset()
    if mode == tf.estimator.ModeKeys.TRAIN:
      labels_dataset = self.labels_inputter.build_dataset()
      dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))
    else:
      dataset = features_dataset
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    return iterator.get_next()

  def serving_input_receiver_fn(self):
    if not self.features_inputter:
      raise NotImplementedError()
    return self.features_inputter.get_serving_input_receiver()

  @abc.abstractmethod
  def _build(self, features, lables, params, mode, config):
    raise NotImplementedError()

  @abc.abstractmethod
  def _initializer(self, params):
    """Return global initializer for this model."""
    raise NotImplementedError()

  @abc.abstractmethod
  def _optimize(self, loss, params):
    raise NotImplementedError()

  @abc.abstractmethod
  def _compute_metrics(self, features, labels, predictions):
    raise NotImplementedError()

  @abc.abstractmethod
  def _compute_loss(self, features, labels, outputs, params, mode):
    raise NotImplementedError()
