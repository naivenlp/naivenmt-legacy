import abc


class EvaluationConfigInterface(abc.ABC):

  @abc.abstractmethod
  def get_eval_batch_size(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_eval_prefetch_buffer_size(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_eval_delay(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_save_eval_predictions(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_external_evaluators(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_exporters(self):
    raise NotImplementedError()
