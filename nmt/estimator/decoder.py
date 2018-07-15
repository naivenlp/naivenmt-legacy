import abc


class DecoderInterface(abc.ABC):

  @abc.abstractmethod
  def decode(self, data, configs, mode):
    raise NotImplementedError()

  @abc.abstractmethod
  def dynamic_decode(self,
                     scope,
                     mode,
                     vocab_size,
                     initial_state,
                     memory,
                     memory_seq_len,
                     dtype,
                     return_alignment_history=True):
    raise NotImplementedError()

  @abc.abstractmethod
  def dynamic_decode_and_search(self,
                                scope,
                                mode,
                                vocab_size,
                                initial_state,
                                beam_width,
                                length_penalty,
                                memory,
                                memory_seq_len,
                                dtype,
                                return_alignment_history=True
                                ):
    raise NotImplementedError()
