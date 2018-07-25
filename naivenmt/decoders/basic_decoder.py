import tensorflow as tf

from naivenmt import utils
from naivenmt.decoders.abstract_decoder import AbstractDecoder


class BasicDecoder(AbstractDecoder):

  def _build_decoder_cell(self,
                          mode,
                          params,
                          encoder_outputs,
                          encoder_state,
                          sequence_length):
    if params.attention:
      raise ValueError("Basic model does not support attention.")

    cell = self._create_rnn_cell(mode, params)
    if mode == tf.estimator.ModeKeys.PREDICT:
      decoder_initial_state = tf.contrib.seq2seq.tile_batch(
        encoder_state, multiplier=params.beam_width)
    else:
      decoder_initial_state = encoder_state
    return cell, decoder_initial_state

  def _create_rnn_cell(self, mode, params):
    if self.extra_args:
      single_cell_fn = self.extra_args.single_cell_fn
    else:
      single_cell_fn = None

    if not single_cell_fn:
      single_cell_fn = self._single_cell

    cell_list = []
    for i in range(params.num_layers):
      residual_conn = (i >= params.num_layers - params.num_resisual_layers)
      single_cell = single_cell_fn(
        unit_type=params.unit_type,
        num_units=params.num_units,
        forget_bias=params.forget_bias,
        dropout=params.dropout,
        mode=mode,
        residual_connection=residual_conn,
        device_str=utils.get_device_str(i, params.num_gpus))
      cell_list.append(single_cell)
    if len(cell_list) == 1:
      return cell_list[0]
    else:
      return tf.contrib.rnn.MultiRNNCell(cell_list)

  def _single_cell(self,
                   unit_type,
                   num_units,
                   forget_bias,
                   dropout,
                   mode,
                   residual_connection=False,
                   device_str=None,
                   residual_fn=None):
    dropout = dropout if mode == tf.estimator.ModeKeys.TRAIN else 0.0

    if unit_type == "lstm":
      single_cell = tf.contrib.rnn.BasicLSTMCell(
        num_units, forget_bias=forget_bias)
    elif unit_type == "gru":
      single_cell = tf.contrib.rnn.GRUCell(num_units)
    elif unit_type == "layer_norm_lstm":
      single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
        num_units, forget_bias=forget_bias, layer_norm=True)
    elif unit_type == "nas":
      single_cell = tf.contrib.rnn.NASCell(num_units)
    else:
      raise ValueError("Invalid unit type: %s" % unit_type)

    if dropout > 0.0:
      single_cell = tf.contrib.rnn.DropoutWrapper(
        cell=single_cell, input_keep_prob=(1.0 - dropout))
    if residual_connection:
      single_cell = tf.contrib.rnn.ResidualWrapper(
        single_cell, residual_fn=residual_fn)

    if device_str:
      single_cell = tf.contrib.rnn.DeviceWrapper(
        single_cell, device_str)

    return single_cell
