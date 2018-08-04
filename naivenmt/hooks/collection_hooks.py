import tensorflow as tf

from naivenmt.hooks.model_tensors_hooks import ModelTensorsHook


class TensorsCollectionHook(ModelTensorsHook):
  """Add tensors to graph's collection."""

  def __init__(self):
    self.tensors = {}

  def on_global_steps_created(self, global_steps):
    # do nothing, cause global steps has been added to collections
    pass

  def on_learning_rate_created(self, learning_rate):
    self.tensors["learning_rate"] = learning_rate

  def on_gradients_created(self, gradients):
    self.tensors["gradients"] = gradients

  def on_gradients_clipped(self, clipped_grads, grad_norm):
    self.tensors["clipped_grads"] = clipped_grads
    self.tensors["grad_norm"] = grad_norm

  def on_optimizer_created(self, optimizer):
    # we do not collect optimizer
    pass

  def on_loss_created(self, loss):
    self.tensors["loss"] = loss

  def end(self):
    for k, v in self.tensors.items():
      tf.add_to_collection(k, v)
    tf.logging.info("Tensors are added to collections.")
