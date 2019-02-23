import tensorflow as tf

from nmt.decoders.basic_decoder import BasicRNNDecoder


class AttentionDecoder(BasicRNNDecoder):

    def _build_cell_and_initial_state(self, mode, encoder_output, encoder_state, sequence_length, params):
        cell = self._build_cell(
            num_layers=params['num_decoder_layers'],
            num_residual_layers=params['num_decoder_residual_layers'],
            num_units=params['num_units'],
            unit_type=params['unit_type'],
            dropout=params['dropout'] if mode == tf.estimator.ModeKeys.TRAIN else 0.0,
            forget_bias=params['forget_bias'],
            residual_fn=None)
        batch_size = tf.size(sequence_length)
        if mode == tf.estimator.ModeKeys.PREDICT and params['beam_width'] > 0:
            encoder_output = tf.contrib.seq2seq.tile_batch(encoder_output, multiplier=params['beam_width'])
            encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=params['beam_width'])
            sequence_length = tf.contrib.seq2seq.tile_batch(sequence_length, multiplier=params['beam_width'])
            batch_size = batch_size * params['beam_width']

        attention_mechanism = self._build_attention_mechanism(
            memory=encoder_output,
            sequence_length=sequence_length,
            params=params)
        alignment_history = (mode == tf.estimator.ModeKeys.PREDICT and params['beam_width'] == 0)
        cell = tf.contrib.seq2seq.AttentionWrapper(
            cell,
            attention_mechanism,
            attention_layer_size=params['num_units'],
            alignment_history=alignment_history,
            output_attention=params['output_attention'],
            name="attention")
        if params['pass_hidden_state']:
            initial_state = cell.zero_state(batch_size, self.dtype).clone(encoder_state)
        else:
            initial_state = cell.zero_state(batch_size, self.dtype)
        return cell, initial_state

    def _build_attention_mechanism(self, memory, sequence_length, params):
        option = params['attention']
        num_units = params['num_units']
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

    def default_config(self):
        config = super(AttentionDecoder, self).default_config()
        config.update({
            "beam_width": 5,
            "attention": "luong",
            "attention_architecture": "standard",
            "sampling_temperature": 0.0,
            "output_attention": True,
            "pass_hidden_state": True
        })
        return config
