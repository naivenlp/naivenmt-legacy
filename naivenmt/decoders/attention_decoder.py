import tensorflow as tf

from naivenmt.decoders.basic_decoder import BasicDecoder


class AttentionDecoder(BasicDecoder):
  """Standard attention decoder."""

  def __init__(self,
               params,
               embedding,
               sos,
               eos,
               scope="decoder",
               dtype=tf.float32,
               single_cell_fn=None,
               attention_mechanism_fn=None):
    super().__init__(params=params,
                     embedding=embedding,
                     sos=sos,
                     eos=eos,
                     scope=scope,
                     dtype=dtype,
                     single_cell_fn=single_cell_fn)

    assert params.attention

    self.attention = params.attention
    self.attention_architecture = params.attention_architecture
    self.output_attention = params.output_attention
    self.pass_hidden_state = params.pass_hidden_state

    self.attention_mechanism_fn = attention_mechanism_fn
    if not self.attention_mechanism_fn:
      self.attention_mechanism_fn = self._attention_mechanism_fn

  def _build_decoder_cell(self,
                          mode,
                          encoder_outputs,
                          encoder_state,
                          sequence_length):
    attention_option = self.attention
    attention_architecture = self.attention_architecture
    assert attention_architecture == "standard"

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

    attention_mechanism = self.attention_mechanism_fn(
      attention_option, self.num_units, memory, sequence_length)

    cell = self._create_rnn_cell(mode)
    alignment_history = (
            mode == tf.estimator.ModeKeys.PREDICT and self.beam_width == 0)
    cell = tf.contrib.seq2seq.AttentionWrapper(
      cell,
      attention_mechanism,
      attention_layer_size=self.num_units,
      alignment_history=alignment_history,
      output_attention=self.output_attention,
      name="attention")
    device = self._get_device_str(self.num_decoder_layers - 1, self.num_gpus)
    cell = tf.contrib.rnn.DeviceWrapper(cell, device)

    if self.pass_hidden_state:
      decoder_initial_state = cell.zero_state(batch_size, self.dtype).clone(
        cell_state=encoder_state)
    else:
      decoder_initial_state = cell.zero_state(batch_size, self.dtype)

    return cell, decoder_initial_state

  @staticmethod
  def _attention_mechanism_fn(option,
                              num_units,
                              memory,
                              sequence_length):
    """Create attention mechanism.

    Args:
      option: attention option
      num_units: number of units
      memory: encoder's outputs
      sequence_length: source sequence length
    """
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
