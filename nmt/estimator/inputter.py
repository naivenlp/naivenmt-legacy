import abc

import tensorflow as tf


class Inputter(abc.ABC):

  def __init__(self, config, dtype):
    self.config = config
    self.dtype = dtype

  @abc.abstractmethod
  def dataset(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def vocab_file(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def vocab(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def vocab_reverse(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def vocab_size(self):
    raise NotImplementedError()


class FeaturesInputterInterface(Inputter):

  def __init__(self, features_file, config, dtype):
    super().__init__(config=config, dtype=dtype)
    self.features_file = features_file

  @abc.abstractmethod
  def features_length(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def serving_input_receiver_fn(self):
    raise NotImplementedError()


class LabelsInputterInterface(Inputter):

  def __init__(self, labels_file, config, dtype):
    super().__init__(config=config, dtype=dtype)
    self.labels_file = labels_file

  @abc.abstractmethod
  def labels_length(self):
    raise NotImplementedError()


class FeaturesInputter(FeaturesInputterInterface):

  def __init__(self, features_file, config, dtype):
    super().__init__(features_file=features_file, config=config, dtype=dtype)

  def serving_input_receiver_fn(self):
    pass

  def dataset(self):
    pass

  def vocab_file(self):
    pass

  def vocab(self):
    pass

  def vocab_reverse(self):
    pass

  def vocab_size(self):
    pass

  def features_length(self):
    pass


class LabelsInputter(LabelsInputterInterface):

  def __init__(self, labels_file, config, dtype):
    super().__init__(labels_file=labels_file, config=config, dtype=dtype)

  def dataset(self):
    pass

  def vocab_file(self):
    pass

  def vocab(self):
    pass

  def vocab_reverse(self):
    pass

  def vocab_size(self):
    pass

  def labels_length(self):
    pass
