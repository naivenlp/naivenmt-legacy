import tensorflow as tf

from naivenmt import utils
from naivenmt.decoders.basic_decoder import BasicDecoder


# TODO(luozhouyang) collect infer summary
class AttentionDecoder(BasicDecoder):

  def _build_decoder_cell(self,
                          mode,
                          params,
                          encoder_outputs,
                          encoder_state,
                          sequence_length):
    attention_option = params.attention
    attention_architecture = params.attention_architecture
    assert attention_architecture == "standard"

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

    cell = self._create_rnn_cell(mode, params)
    alignment_history = (
            mode == tf.estimator.ModeKeys.PREDICT and params.beam_width == 0)
    cell = tf.contrib.seq2seq.AttentionWrapper(
      cell,
      attention_mechanism,
      attention_layer_size=params.num_units,
      alignment_history=alignment_history,
      output_attention=params.output_attention,
      name="attention")
    device = utils.get_device_str(params.num_decoder_layers - 1,
                                  params.num_gpus)
    cell = tf.contrib.rnn.DeviceWrapper(cell, device)

    dtype = tf.float32
    if params.pass_hidden_state:
      decoder_initial_state = cell.zero_state(batch_size, dtype).clone(
        cell_state=encoder_state)
    else:
      decoder_initial_state = cell.zero_state(batch_size, dtype)

    return cell, decoder_initial_state

  def _create_attention_mechanism_fn(self):
    def _create_attention_mechanism_fn(option, num_units,
                                       memory, sequence_length):
      if option == "luong":
        mechanism = tf.contrib.seq2seq.LuongAttention(
          num_units, memory, memory_sequence_length=sequence_length)
      elif option == "scaled_luong":
        mechanism = tf.contrib.seq2seq.LuongAttention(
          num_units, memory, memory_sequence_length=sequence_length, scale=True)
      elif option == "bahdanau":
        mechanism = tf.contrib.seq2seq.BahdanauAttention(
          num_units, memory, memory_sequence_length=sequence_length)
      elif option == "normed_bahdanau":
        mechanism = tf.contrib.seq2seq.BahdanauAttention(
          num_units,
          memory,
          memory_sequence_length=sequence_length,
          normalize=True)
      else:
        raise ValueError("Invalid attention option: %s" % option)

      return mechanism

    return _create_attention_mechanism_fn
