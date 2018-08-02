import tensorflow as tf


class CountParamsHook(tf.train.SessionRunHook):
  """Logs the number of trainable parameters."""

  def begin(self):
    total = 0
    for variable in tf.trainable_variables():
      shape = variable.get_shape()
      count = 1
      for dim in shape:
        count *= dim.value
      total += count
    tf.logging.info("Number of trainable parameters: %d", total)
