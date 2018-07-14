import abc


class DataConfigInterface(abc.ABC):

  @abc.abstractmethod
  def training_features_file(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def training_labels_file(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def eval_features_file(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def eval_labels_file(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def features_vocab_file(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def labels_vocab_file(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def max_features_len(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def max_labels_len(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def out_dir(self):
    raise NotImplementedError()
