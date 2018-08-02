import abc


class ModelLifecycleHook(abc.ABC):
  """Listeners to listen model's lifecycle.

  Generally speaking, we can just get information from tensors,
  but not to update these tensors.
  """

  @abc.abstractmethod
  def before_encode(self, mode, features):
    """Listen tensors states before encode.

    Args:
      mode: mode, one of estimator's ModeKeys.
      features: features inputs, instance of ``naivenmt.inputters.Features``.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def after_encode(self, mode, features, outputs, state):
    """Listen tensors states after encode, before decode.

    Args:
      mode: mode, one of estimator's ModeKeys.
      features: features inputs, instance of ``naivenmt.inputters.Features``.
      outputs: encoder's outputs.
      state: encoder's output state.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def before_decode(self, mode, outputs, state, labels, src_seq_len):
    """Listen tensors state before decode.

    Args:
      mode: mode, one of estimator's ModeKeys.
      outputs: encoder's outputs
      state: encoder's output state
      labels: labels inputs, instance of ``naivenmt.inputters.Labels``.
      src_seq_len: source sequence length
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def after_decode(self, mode, logits, loss, sample_id, final_context_state):
    """Listen tensors state after decode.

    Args:
      mode: mode, one of estimator's ModeKeys.
      logits: logits
      loss: loss, None for PREDICT mode.
      sample_id: sample id results of decode.
      final_context_state: state of decoder.
    """
    raise NotImplementedError()
