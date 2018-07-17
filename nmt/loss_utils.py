import tensorflow as tf


def _smooth_one_hot_labels(logits, labels, label_smoothing):
  label_smoothing = tf.constant(label_smoothing, dtype=logits.dtype)
  num_classes = tf.shape(logits)[-1]
  return tf.one_hot(
    tf.cast(labels, tf.int32),
    num_classes,
    on_value=1.0 - label_smoothing,
    off_value=label_smoothing / tf.cast(num_classes - 1, label_smoothing.dtype),
    dtype=logits.dtype)


def _softmax_cross_entropy(logits,
                           labels,
                           smoothing,
                           mode):
  if mode == tf.estimator.ModeKeys.TRAIN and smoothing > 0.0:
    smoothed_labels = _smooth_one_hot_labels(logits, labels, smoothing)
    if hasattr(tf.nn, "softmax_cross_entropy_with_logits_v2"):
      cross_entropy_fn = tf.nn.softmax_cross_entropy_with_logits_v2
    else:
      cross_entropy_fn = tf.nn.softmax_cross_entropy_with_logits
    return cross_entropy_fn(
      logits=logits, labels=smoothed_labels)
  else:
    return tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels)


def cross_entropy_sequence_loss(logits,
                                labels,
                                sequence_length,
                                average_in_time=False,
                                smoothing=0.0,
                                mode=tf.estimator.ModeKeys.TRAIN):
  batch_size = tf.shape(logits)[0]
  max_time = tf.shape(logits)[1]
  cross_entropy = _softmax_cross_entropy(logits, labels, smoothing, mode)
  weights = tf.sequence_mask(sequence_length, maxlen=max_time,
                             dtype=cross_entropy.dtype)
  loss = tf.reduce_sum(cross_entropy * weights)
  loss_token_normalizer = tf.reduce_sum(weights)
  if average_in_time or mode != tf.estimator.ModeKeys.TRAIN:
    loss_normalizer = loss_token_normalizer
  else:
    loss_normalizer = tf.cast(batch_size, loss.dtype)

  return loss, loss_normalizer, loss_token_normalizer
