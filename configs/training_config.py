import abc


class TrainingConfigInterface(abc.ABC):

  @abc.abstractmethod
  def get_training_batch_size(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_save_checkpoints_steps(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_keep_max_checkpoints(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_save_summary_steps(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_training_steps(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_bucket_width(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_shuffle_buffer_size(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_training_prefetch_buffer_size(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_avg_last_checkpoints(self):
    raise NotImplementedError()
