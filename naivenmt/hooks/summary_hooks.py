from naivenmt.hooks import ModelTensorsHook
import tensorflow as tf


class TensorSummaryHook(ModelTensorsHook):
  """Summary tensors."""

  def __init__(self):
    self.summaries = {}

  def on_global_steps_created(self, global_steps):
    self.summaries["global_steps"] = global_steps

  def on_learning_rate_created(self, learning_rate):
    self.summaries["learning_rate"] = learning_rate

  def on_gradients_created(self, gradients):
    self.summaries["gradients"] = gradients

  def on_gradients_clipped(self, clipped_grads, grad_norm):
    self.summaries["clipped_grads"] = tf.global_norm(clipped_grads)
    self.summaries["grad_norm"] = grad_norm

  def on_optimizer_created(self, optimizer):
    pass

  def on_loss_created(self, loss):
    if not loss:  # predict mode
      return
    # TODO(luozhouyang) need a `mode` arg?
    self.summaries["loss"] = loss

  def end(self):
    summaries = []
    for k, v in self.summaries.items():
      summary = tf.summary.scalar(k, v)
      summaries.append(summary)
    tf.summary.merge(summaries)
    tf.logging.info("Tensors summaries are merged.")
