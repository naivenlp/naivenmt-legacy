import abc


class TrainingConfigInterface(abc.ABC):

  @abc.abstractmethod
  def training_batch_size(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def save_checkpoints_steps(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def keep_max_checkpoints(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def save_summary_steps(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def training_steps(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def bucket_width(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def shuffle_buffer_size(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def training_prefetch_buffer_size(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def avg_last_checkpoints(self):
    raise NotImplementedError()
