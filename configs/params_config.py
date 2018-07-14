import abc


class ParamsConfigInterface(abc.ABC):

  @abc.abstractmethod
  def get_optimizer(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_learning_rate(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_init_value(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_clip_gradients(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_regularization(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_avg_loss_in_time(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_decay_type(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_decay_rate(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_decay_steps(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_start_decay_steps(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_min_learning_rate(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_beam_width(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_replace_unknown_target(self):
    raise NotImplementedError()
