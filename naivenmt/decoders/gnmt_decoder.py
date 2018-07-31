import tensorflow as tf
from tensorflow.python.util import nest

from naivenmt.decoders.attention_decoder import AttentionDecoder
from naivenmt.decoders.gnmt_attention_multi_cell import GNMTAttentionMultiCell


class GNMTDecoder(AttentionDecoder):

  def __init__(self,
               params,
               embedding,
               sos,
               eos,
               scope="decoder",
               dtype=tf.float32,
               single_cell_fn=None,
               attention_mechanism_fn=None,
               residual_fn=None):
    super().__init__(params=params,
                     embedding=embedding,
                     sos=sos,
                     eos=eos,
                     scope=scope,
                     dtype=dtype,
                     single_cell_fn=single_cell_fn,
                     attention_mechanism_fn=attention_mechanism_fn)

    self.residual_fn = residual_fn
    if not self.residual_fn:
      self.residual_fn = self._residual_fn

  def _build_decoder_cell(self,
                          mode,
                          encoder_outputs,
                          encoder_state,
                          sequence_length):
    attention_option = self.attention
    attention_architecture = self.attention_architecture
    if attention_architecture == "standard":
      return super()._build_decoder_cell(
        mode, encoder_outputs, encoder_state, sequence_length)

    if self.time_major:
      memory = tf.transpose(encoder_outputs, [1, 0, 2])
    else:
      memory = encoder_outputs

    if mode == tf.estimator.ModeKeys.PREDICT and self.beam_width > 0:
      memory = tf.contrib.seq2seq.tile_batch(
        memory, multiplier=self.beam_width)
      sequence_length = tf.contrib.seq2seq.tile_batch(
        sequence_length, multiplier=self.beam_width)
      encoder_state = tf.contrib.seq2seq.tile_batch(
        encoder_state, multiplier=self.beam_width)
      batch_size = tf.size(sequence_length) * self.beam_width
    else:
      batch_size = tf.size(sequence_length)

    attention_mechanism = self._attention_mechanism_fn(
      attention_option, self.num_units, memory, sequence_length)

    cell_list = self._cell_list(
      self.unit_type,
      self.num_units,
      self.num_decoder_layers,
      self.num_decoder_residual_layers,
      self.forget_bias, self.dropout,
      mode,
      self.num_gpus,
      single_cell_fn=self.single_cell_fn,
      residual_fn=self.residual_fn)
    attention_cell = cell_list.pop(0)

    alignment_history = (
            mode == tf.estimator.ModeKeys.PREDICT and self.beam_width == 0)
    attention_cell = tf.contrib.seq2seq.AttentionWrapper(
      attention_cell,
      attention_mechanism,
      attention_layer_size=self.num_units,
      alignment_history=alignment_history,
      output_attention=self.output_attention,
      name="attention")
    if attention_architecture == "gnmt":
      cell = GNMTAttentionMultiCell(attention_cell, cell_list)
    elif attention_architecture == "gnmt_v2":
      cell = GNMTAttentionMultiCell(attention_cell, cell_list,
                                    use_new_attention=True)
    else:
      raise ValueError(
        "Invalid attention architecture: %s" % attention_architecture)
    if self.pass_hidden_state:
      decoder_initial_state = tuple(
        zs.clone(cell_state=es)
        if isinstance(zs, tf.contrib.seq2seq.AttentionWrapperState) else es
        for zs, es in zip(
          cell.zero_state(batch_size, self.dtype), encoder_state))
    else:
      decoder_initial_state = cell.zero_state(batch_size, self.dtype)

    return cell, decoder_initial_state

  @staticmethod
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
