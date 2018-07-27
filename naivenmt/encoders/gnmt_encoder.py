import tensorflow as tf

from naivenmt.encoders.basic_encoder import BasicEncoder
from naivenmt import utils


class GNMTEncoder(BasicEncoder):

  def encode(self, mode, features, sequence_length, params, configs):
    if params.encoder_type in ["uni", "bi"]:
      return super().encode(mode, features, sequence_length, params, configs)

    if params.encoder_type != "gnmt":
      raise ValueError("Invalid encoder type: %s" % params.encoder_type)

    num_bi_layers = 1
    num_uni_layers = params.num_encoder_layers - num_bi_layers

    with tf.variable_scope("encoder") as scope:
      # TODO(luozhouyang) handle dtype
      dtype = scope.dtype

      # TODO(luozhouyang) make sure features is a Tensor
      bi_encoder_outputs, bi_encoder_state = self._build_bidirectional_rnn(
        mode=mode,
        inputs=self.embedding.encoder_embedding_input(features),
        sequence_length=sequence_length,
        dtype=dtype,
        params=params,
        num_bi_layers=num_bi_layers,
        num_bi_residual_layers=0)

      uni_cell = utils.create_rnn_cells(
        unit_type=params.unit_type,
        num_units=params.num_units,
        num_layers=num_uni_layers,
        num_residual_layers=params.num_encoder_residual_layers,
        forget_bias=params.forget_bias,
        dropout=params.dropout,
        mode=mode,
        num_gpus=params.num_gpus,
        base_gpu=1,
        single_cell_fn=self._single_cell_fn)

      encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
        uni_cell,
        bi_encoder_outputs,
        dtype=dtype,
        sequence_length=sequence_length,
        time_major=params.time_major)

      encoder_state = (bi_encoder_state[1],) + (
        (encoder_state,) if num_uni_layers == 1 else encoder_state)

      return encoder_outputs, encoder_state
