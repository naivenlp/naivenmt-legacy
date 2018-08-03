import abc

import tensorflow as tf


class EncoderInterface(abc.ABC):
  """Encoder interface."""

  @abc.abstractmethod
  def encode(self, mode, features):
    """Encode source inputs.

    Args:
      mode: mode
      features: source inputs, an instance of ``naivenmt.inputters.Features``

    Returns:
      encoder_outputs: outputs of encoder
      encoder_state: state of encoder
    """
    raise NotImplementedError()


class AbstractEncoder(EncoderInterface):
  """Abstract encoder."""

  def __init__(self,
               params,
               embedding,
               scope=None,
               dtype=None):
    """Init abstract encoder.

    Args:
      params: hparams
      embedding: embedding, an instance of ``naivenmt.embeddings.Embedding``
      scope: variables scope
      dtype: variables dtype
    """

    self.embedding = embedding
    self.scope = scope or "encoder"
    self.dtype = dtype or tf.float32

    self.num_encoder_layers = params.num_encoder_layers
    self.num_encoder_residual_layers = params.num_encoder_residual_layers
    self.encoder_type = params.encoder_type
    self.time_major = params.tima_major
    self.unit_type = params.unit_type
    self.num_units = params.num_units
    self.forget_bias = params.forget_bias
    self.dropout = params.dropout
    self.num_gpus = params.num_gpus

  def encode(self, mode, features):
    num_layers = self.num_encoder_layers
    num_residual_layers = self.num_encoder_residual_layers
    sequence_length = features.source_sequence_length
    with tf.variable_scope(self.scope, dtype=self.dtype) as scope:
      encoder_embedding_input = self.embedding.encoder_embedding_input(
        features.source_ids)

      if self.encoder_type == "uni":
        cell = self._build_encoder_cell(mode, num_layers, num_residual_layers)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
          cell,
          encoder_embedding_input,
          dtype=self.dtype,
          sequence_length=sequence_length,
          time_major=self.time_major,
          swap_memory=True)
      elif self.encoder_type == "bi":
        num_bi_layers = int(num_layers / 2)
        num_bi_residual_layers = int(num_residual_layers / 2)

        encoder_outputs, bi_encoder_state = self._build_bidirectional_rnn(
          mode=mode,
          inputs=encoder_embedding_input,
          sequence_length=sequence_length,
          num_bi_layers=num_bi_layers,
          num_bi_residual_layers=num_bi_residual_layers)

        if num_bi_layers == 1:
          encoder_state = bi_encoder_state
        else:
          encoder_state = []
          for layer_id in range(num_bi_layers):
            encoder_state.append(bi_encoder_state[0][layer_id])
            encoder_state.append(bi_encoder_state[1][layer_id])
          encoder_state = tuple(encoder_state)
      else:
        raise ValueError("Invalid encoder type: %s" % self.encoder_type)
      return encoder_outputs, encoder_state

  @abc.abstractmethod
  def _build_encoder_cell(self,
                          mode,
                          num_layers,
                          num_residual_layers,
                          base_gpu=0):
    """Create encoder cells.

    Args:
      mode: mode
      num_layers: number of layers
      num_residual_layers: number of residual layers
      base_gpu: offset of gpu device, to decide which gpu to use

    Returns:
      encoder network with rnn cells.
    """
    raise NotImplementedError()

  def _build_bidirectional_rnn(self,
                               mode,
                               inputs,
                               sequence_length,
                               num_bi_layers,
                               num_bi_residual_layers,
                               base_gpu=0):
    """Create bi-directional cells.

    Args:
      mode: mode
      inputs: embedding inputs
      sequence_length: source sequence length
      num_bi_layers: number of bidirectional layers
      num_bi_residual_layers: number of bidirectional residual layers
      base_gpu: offset of gpu device, to decide which gpu to use.

    Returns:
      encoder network with bidirectional rnn cells.
    """
    forward_cell = self._build_encoder_cell(
      mode, num_bi_layers, num_bi_residual_layers, base_gpu=base_gpu)
    backward_cell = self._build_encoder_cell(
      mode, num_bi_layers, num_bi_residual_layers,
      base_gpu=(base_gpu + num_bi_layers))

    bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
      forward_cell,
      backward_cell,
      inputs,
      dtype=self.dtype,
      sequence_length=sequence_length,
      time_major=self.time_major,
      swap_memory=True)

    return tf.concat(bi_outputs, -1), bi_state
