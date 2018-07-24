import abc


class Decoder(abc.ABC):

  @abc.abstractmethod
  def decode(self, encoder_outputs, encoder_state, labels, params):
    raise NotImplementedError()
