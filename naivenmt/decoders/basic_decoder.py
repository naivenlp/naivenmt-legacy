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

    cell_list = self._cell_list(
      params.unit_type, params.num_units, params.num_decoder_layers,
      # TODO(luozhouyang) add `num_decoder_residual_layers` in hparams
      params.num_decoder_residual_layers, params.forget_bias, params.dropout,
      mode=mode,
      num_gpus=params.num_gpus,
      single_cell_fn=self._single_cell_fn,
      residual_fn=self._residual_fn)
    if len(cell_list) == 1:
      return cell_list[0]
    else:
      return tf.contrib.rnn.MultiRNNCell(cell_list)

  @staticmethod
  def _cell_list(unit_type, num_units, num_layers, num_residual_layers,
                 forget_bias, dropout, mode, num_gpus, base_gpu=0,
                 single_cell_fn=None,
                 residual_fn=None):
    cells = []
    assert single_cell_fn
    for i in range(num_layers):
      residual_conn = (i >= num_layers - num_residual_layers)
      device_str = utils.get_device_str(i + base_gpu, num_gpus)
      single_cell = single_cell_fn(
        unit_type, num_units, forget_bias, dropout, mode, residual_conn,
        device_str, residual_fn)
      cells.append(single_cell)
    return cells

  def _create_single_cell_fn(self):
    def _single_cell_fn(unit_type, num_units, forget_bias, dropout, mode,
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

    return _single_cell_fn
