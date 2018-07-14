import abc


class EvaluationConfigInterface(abc.ABC):

  @abc.abstractmethod
  def eval_batch_size(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def eval_prefetch_buffer_size(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def eval_delay(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def save_eval_predictions(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def external_evaluators(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def exporters(self):
    raise NotImplementedError()
