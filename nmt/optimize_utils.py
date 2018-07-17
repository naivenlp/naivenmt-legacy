import tensorflow as tf


def _learning_rate_warm_up(params, lr):
  raise NotImplementedError()


def _learning_rate_decay(params, lr):
  raise NotImplementedError()


def _optimizer(params):
  lr = params.learning_rate
  opt_name = params.optimizer
  if opt_name == "sgd":
    opt = tf.train.GradientDescentOptimizer(lr)
    tf.summary.scalar("lr", lr)
  elif opt_name == "adam":
    opt = tf.train.AdamOptimizer(lr)
  else:
    raise ValueError("You must specify a optimizer.")
  return opt


def _gradient_clip(grads, max_gradient_norm):
  raise NotImplementedError()


def optimize(loss, params):
  global_step = tf.train.get_or_create_global_step()
  lr = tf.constant(params.learning_rate)
  lr = _learning_rate_warm_up(params, lr)
  lr = _learning_rate_decay(params, lr)
  opt = _optimizer(params)
  gradients = tf.gradients(loss, tf.trainable_variables(),
                           colocate_gradients_with_ops=True)
  clipped_grads, grad_norm_summary, grad_norm = _gradient_clip(
    gradients, max_gradient_norm=params.clip_gradients)
  tf.summary.merge([tf.summary.scalar("lr", lr),
                    tf.summary.scalar("train_loss", loss)] + grad_norm_summary)
  return opt.apply_gradients(
    zip(clipped_grads, tf.trainable_variables()),
    global_step=global_step)
