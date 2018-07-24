import abc


class Encoder(abc.ABC):

  @abc.abstractmethod
  def encode(self, features, labels):
    raise NotImplementedError()
