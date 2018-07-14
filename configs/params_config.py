import abc


class ParamsConfigInterface(abc.ABC):

  @abc.abstractmethod
  def optimizer(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def learning_rate(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def init_value(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def clip_gradients(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def regularization(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def avg_loss_in_time(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def decay_type(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def decay_rate(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def decay_steps(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def start_decay_steps(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def min_learning_rate(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def beam_width(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def replace_unknown_target(self):
    raise NotImplementedError()
