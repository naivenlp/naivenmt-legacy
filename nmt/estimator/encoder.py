import abc


class EncoderInterface(abc.ABC):

  @abc.abstractmethod
  def encode(self, data, configs, mode):
    raise NotImplementedError()


