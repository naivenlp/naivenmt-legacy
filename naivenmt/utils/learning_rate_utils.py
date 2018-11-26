import tensorflow as tf


def learning_rate_warmup(lr, params):
  warmup_steps = params.warmup_steps
  warmup_scheme = params.warmup_scheme
  global_steps = tf.train.get_or_create_global_step()
  if warmup_scheme == "t2t":
    warmup_factor = tf.exp(tf.log(0.01) / warmup_steps)
    inv_decay = warmup_factor ** (tf.to_float(warmup_steps - global_steps))
  elif not warmup_scheme:
    raise ValueError("Unknown warmup scheme %s" % warmup_scheme)
  else:
    warmup_steps = 0
  return tf.cond(global_steps < warmup_steps,
                 lambda: inv_decay * lr,
                 lambda: lr,
                 name="lr_warmup_cond")


def learning_rate_decay(lr, params):
  decay_scheme = params.decay_scheme
  num_train_steps = params.num_train_steps
  start_decay_steps = num_train_steps
  decay_steps = 0
  decay_factor = 1.0
  if decay_scheme in ['luong5', 'luong10', 'luong234']:
    decay_factor = 0.5
    if decay_scheme == "luong5":
      start_decay_steps = int(num_train_steps / 2)
      decay_times = 5
    elif decay_scheme == "luong10":
      start_decay_steps = int(num_train_steps / 2)
      decay_times = 10
    else:
      start_decay_steps = int(num_train_steps * 2 / 3)
      decay_times = 4
    remain_steps = num_train_steps - start_decay_steps
    decay_steps = int(remain_steps / decay_times)
  elif decay_scheme:
    raise ValueError("Unknown decay scheme %s" % decay_scheme)
  global_steps = tf.train.get_or_create_global_step()
  return tf.cond(global_steps < start_decay_steps,
                 lambda: lr,
                 lambda: tf.train.exponential_decay(
                   lr,
                   (global_steps - start_decay_steps),
                   decay_steps,
                   decay_factor,
                   staircase=True),
                 name="lr_decay_cond")
