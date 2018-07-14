import abc


class InferenceConfigInterface(abc.ABC):

  @abc.abstractmethod
  def get_infer_batch_size(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_infer_prefetch_size(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_n_best(self):
    raise NotImplementedError()
