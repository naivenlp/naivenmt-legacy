import abc


class InferenceConfigInterface(abc.ABC):

  @abc.abstractmethod
  def infer_batch_size(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def infer_prefetch_size(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def n_best(self):
    raise NotImplementedError()
