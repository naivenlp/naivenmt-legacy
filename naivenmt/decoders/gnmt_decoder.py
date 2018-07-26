import tensorflow as tf
from tensorflow.python.util import nest

from naivenmt.decoders.attention_decoder import AttentionDecoder
from naivenmt.decoders.gnmt_attention_multi_cell import GNMTAttentionMultiCell


class GNMTDecoder(AttentionDecoder):

  def _build_decoder_cell(self,
                          mode,
                          params,
                          encoder_outputs,
                          encoder_state,
                          sequence_length):
    attention_option = params.attention
    attention_architecture = params.attention_architecture
    if attention_architecture == "standard":
      return super()._build_decoder_cell(
        mode, params, encoder_outputs, encoder_state, sequence_length)

    if params.time_major:
      memory = tf.transpose(encoder_outputs, [1, 0, 2])
    else:
      memory = encoder_outputs

    if mode == tf.estimator.ModeKeys.PREDICT and params.beam_width > 0:
      memory = tf.contrib.seq2seq.tile_batch(
        memory, multiplier=params.beam_width)
      sequence_length = tf.contrib.seq2seq.tile_batch(
        sequence_length, multiplier=params.beam_width)
      encoder_state = tf.contrib.seq2seq.tile_batch(
        encoder_state, multiplier=params.beam_width)
      batch_size = tf.size(sequence_length) * params.beam_width
    else:
      batch_size = tf.size(sequence_length)

    attention_mechanism = self._attention_mechanism_fn(
      attention_option, params.num_units, memory, sequence_length, mode)

    cell_list = self._cell_list(
      params.unit_type, params.num_units, params.num_decoder_layers,
      params.num_decoder_residual_layers, params.forget_bias, params.dropout,
      mode, params.num_gpus, single_cell_fn=self._single_cell_fn,
      residual_fn=self._residual_fn)
    attention_cell = cell_list.pop(0)

    alignment_history = (
            mode == tf.estimator.ModeKeys.PREDICT and params.beam_width == 0)
    attention_cell = tf.contrib.seq2seq.AttentionWrapper(
      attention_cell,
      attention_mechanism,
      attention_layer_size=params.num_units,
      alignment_history=alignment_history,
      output_attention=params.output_attention,
      name="attention")
    if attention_architecture == "gnmt":
      cell = GNMTAttentionMultiCell(attention_cell, cell_list)
    elif attention_architecture == "gnmt_v2":
      cell = GNMTAttentionMultiCell(attention_cell, cell_list,
                                    use_new_attention=True)
    else:
      raise ValueError(
        "Invalid attention architecture: %s" % attention_architecture)
    dtype = tf.float32
    if params.pass_hidden_state:
      decoder_initial_state = tuple(
        zs.clone(cell_state=es)
        if isinstance(zs, tf.contrib.seq2seq.AttentionWrapperState) else es
        for zs, es in zip(
          cell.zero_state(batch_size, dtype), encoder_state))
    else:
      decoder_initial_state = cell.zero_state(batch_size, dtype)

    return cell, decoder_initial_state

  def _create_residual_fn(self):
    def _residual_fn(inputs, outputs):
      def split_input(inp, out):
        out_dim = out.get_shape().as_list()[-1]
        inp_dim = inp.get_shape().as_list()[-1]
        return tf.split(inp, [out_dim, inp_dim - out_dim], axis=-1)

      actual_inputs, _ = nest.map_structure(split_input, inputs, outputs)

      def assert_shape_match(inp, out):
        inp.get_shape().assert_is_compatible_with(out.get_shape())

      nest.assert_same_structure(actual_inputs, outputs)
      nest.map_structure(assert_shape_match, actual_inputs, outputs)
      return nest.map_structure(lambda inp, out: inp + out, actual_inputs,
                                outputs)

    return _residual_fn
