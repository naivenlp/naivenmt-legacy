import abc

import tensorflow as tf


class EncoderInterface(abc.ABC):

  @abc.abstractmethod
  def encode(self, mode, features, sequence_length, params, configs):
    raise NotImplementedError()


class AbstractEncoder(EncoderInterface):

  def __init__(self, embedding, scope="encoder"):
    self.embedding = embedding
    self.scope = scope
    self._single_cell_fn = self._create_single_cell_fn()

  def encode(self, mode, features, sequence_length, params, configs):
    num_layers = params.num_encoder_layers
    num_residual_layers = params.num_encoder_residual_layers

    # TODO(luozhouyang) handle dtype
    with tf.variable_scope(self.scope) as scope:
      encoder_embedding_input = self.embedding.encoder_embedding_input(features)

      if params.encoder_type == "uni":
        cell = self._build_encoder_cell(mode, params, num_layers,
                                        num_residual_layers)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
          cell,
          encoder_embedding_input,
          dtype=scope.dtype,
          sequence_length=sequence_length,
          time_major=params.time_major,
          swap_memory=True)
      elif params.encoder_type == "bi":
        num_bi_layers = int(num_layers / 2)
        num_bi_residual_layers = int(num_residual_layers / 2)

        encoder_outputs, bi_encoder_state = self._build_bidirectional_rnn(
          mode=mode,
          inputs=encoder_embedding_input,
          sequence_length=sequence_length,
          dtype=scope.dtype,
          params=params,
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
        raise ValueError("Invalid encoder type: %s" % params.encoder_type)
      return encoder_outputs, encoder_state

  @abc.abstractmethod
  def _build_encoder_cell(self, mode, params, num_layers, num_residual_layers,
                          base_gpu=0):
    raise NotImplementedError()

  def _build_bidirectional_rnn(self,
                               mode,
                               inputs,
                               sequence_length,
                               dtype,
                               params,
                               num_bi_layers,
                               num_bi_residual_layers,
                               base_gpu=0):
    forward_cell = self._build_encoder_cell(mode, params, num_bi_layers,
                                            num_bi_residual_layers,
                                            base_gpu=base_gpu)
    backward_cell = self._build_encoder_cell(
      mode, params, num_bi_layers, num_bi_residual_layers,
      base_gpu=(base_gpu + num_bi_layers))

    bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
      forward_cell,
      backward_cell,
      inputs,
      dtype=dtype,
      sequence_length=sequence_length,
      time_major=params.time_major,
      swap_memory=True)

    return tf.concat(bi_outputs, -1), bi_state

  def _create_single_cell_fn(self):
    return None
