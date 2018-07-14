import abc


class DataConfigInterface(abc.ABC):

  @abc.abstractmethod
  def get_training_features_file(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_training_labels_file(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_eval_features_file(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_eval_labels_file(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_features_vocab_file(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_labels_vocab_file(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_max_features_len(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_max_labels_len(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_out_dir(self):
    raise NotImplementedError()
