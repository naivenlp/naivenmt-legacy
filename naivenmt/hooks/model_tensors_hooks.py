import abc


class ModelTensorsHook(abc.ABC):
  """Listen tensors state when it has been created."""

  @abc.abstractmethod
  def on_global_steps_created(self, global_steps):
    """Listen global_steps.

    Args:
      global_steps: global steps.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def on_learning_rate_created(self, learning_rate):
    """Listen learning_rate.

    Args:
      learning_rate: learning rate.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def on_gradients_created(self, gradients):
    """Listen gradients before clipping.

    Args:
      gradients: gradients.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def on_gradients_clipped(self, clipped_grads, grad_norm):
    """Listen states of clipped gradients.

    Args:
      clipped_grads: clipped gradients.
      grad_norm: gradients norm.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def on_optimizer_created(self, optimizer):
    """Listen optimizer.

    Args:
      optimizer: optimizer
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def on_loss_created(self, loss):
    """Listen loss.

    Args:
      loss
    """
    raise NotImplementedError()
