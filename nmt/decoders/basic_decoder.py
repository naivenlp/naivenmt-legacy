import tensorflow as tf

from nmt.decoders.abstract_decoder import EmbeddingDecoder


class BasicRNNDecoder(EmbeddingDecoder):

    def decode(self, outputs, states, labels, src_sequence_len, mode, params):
        default_params = self.default_config()
        if params:
            default_params.update(**params)
        params = default_params

        if params['time_major']:
            outputs = tf.transpose(outputs, perm=[1, 0, 2])

        with tf.variable_scope(self.scope, dtype=self.dtype, reuse=tf.AUTO_REUSE) as scope:
            cell, initial_state = self._build_cell_and_initial_state(mode, outputs, states, src_sequence_len, params)

            if mode != tf.estimator.ModeKeys.PREDICT:
                helper = tf.contrib.seq2seq.TrainingHelper(
                    inputs=labels['tgt_in'],
                    sequence_length=labels['tgt_len'],
                    time_major=params['time_major'])
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=cell,
                    helper=helper,
                    initial_state=initial_state)
                decoder_outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder,
                    output_time_major=False,
                    swap_memory=params['swap_memory'])
                sample_id = decoder_outputs.sample_id
                logits = tf.layers.dense(inputs=decoder_outputs.rnn_output, units=params['target_vocab_size'],
                                         use_bias=False)
                return logits, sample_id, final_context_state

            decoder = self._build_decoder(cell, initial_state, params)
            decoder_outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder,
                maximum_iterations=self._max_infer_iterations(src_sequence_len, params),
                output_time_major=False,
                swap_memory=params['swap_memory'])
            logits = outputs.rnn_output
            sample_id = outputs.sample_id
            return logits, sample_id, final_context_state

    def _build_cell(self,
                    num_layers,
                    num_residual_layers,
                    num_units,
                    unit_type,
                    dropout,
                    forget_bias,
                    residual_fn):
        cells = []
        for i in range(num_layers):
            residual = (i >= num_layers - num_residual_layers)
            if unit_type == "lstm":
                cell = tf.nn.rnn_cell.LSTMCell(num_units=num_units, forget_bias=forget_bias)
            elif unit_type == "layer_norm_lstm":
                cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units, forget_bias, layer_norm=True)
            elif unit_type == "gru":
                cell = tf.nn.rnn_cell.GRUCell(num_units=num_units)
            elif unit_type == "nas":
                cell = tf.contrib.rnn.NASCell(num_units=num_units)
            else:
                raise ValueError("Invalid unit_type: %s" % unit_type)

            if dropout > 0.0:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, 1.0 - dropout)
            if residual and residual_fn:
                cell = tf.nn.rnn_cell.ResidualWrapper(cell, residual_fn)

            cells.append(cell)
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        return cell

    def _build_initial_state(self, mode, encoder_output, encoder_state, sequence_length, params):
        if mode == tf.estimator.ModeKeys.PREDICT and params['beam_width'] > 0:
            initial_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=params['beam_width'])
        else:
            initial_state = encoder_state
        return initial_state

    def _build_cell_and_initial_state(self, mode, encoder_output, encoder_state, sequence_length, params):
        cell = self._build_cell(
            num_layers=params['num_decoder_layers'],
            num_residual_layers=params['num_decoder_residual_layers'],
            num_units=params['num_units'],
            unit_type=params['unit_type'],
            dropout=params['dropout'] if mode == tf.estimator.ModeKeys.TRAIN else 0.0,
            forget_bias=params['forget_bias'],
            residual_fn=None)
        initial_state = self._build_initial_state(mode, encoder_output, encoder_state, sequence_length, params)
        return cell, initial_state

    def _build_decoder(self, cell, initial_state, params):
        if params['beam_width'] > 0:
            return self._build_beam_decoder(cell, initial_state, params)
        if params['sampling_temperature'] > 0.0:
            return self._build_sampling_decoder(cell, initial_state, params)
        return self._build_greedy_decoder(cell, initial_state, params)

    def _build_greedy_decoder(self, cell, initial_state, params):
        sos_id = tf.constant(value=params['sos_id'], dtype=tf.int32)
        start_tokens = tf.fill([params['infer_batch_size']], sos_id)
        end_token = tf.constant(value=params['eos_id'], dtype=tf.int32)

        output_layer = tf.layers.Dense(units=params['target_vocab_size'], use_bias=False)

        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding=self.embedding,
            start_tokens=start_tokens,
            end_token=end_token)
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=cell,
            helper=helper,
            initial_state=initial_state,
            output_layer=output_layer)
        return decoder

    def _build_beam_decoder(self, cell, initial_state, params):
        sos_id = tf.constant(value=params['sos_id'], dtype=tf.int32)
        start_tokens = tf.fill([params['infer_batch_size']], sos_id)
        end_token = tf.constant(value=params['eos_id'], dtype=tf.int32)

        output_layer = tf.layers.Dense(units=params['target_vocab_size'], use_bias=False)

        decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=cell,
            embedding=self.embedding,
            start_tokens=start_tokens,
            end_token=end_token,
            initial_state=initial_state,
            beam_width=params['beam_width'],
            output_layer=output_layer,
            length_penalty_weight=params['length_penalty_weight'])
        return decoder

    def _build_sampling_decoder(self, cell, initial_state, params):
        sos_id = tf.constant(value=params['sos_id'], dtype=tf.int32)
        start_tokens = tf.fill([params['infer_batch_size']], sos_id)
        end_token = tf.constant(value=params['eos_id'], dtype=tf.int32)

        output_layer = tf.layers.Dense(units=params['target_vocab_size'], use_bias=False)

        helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
            embedding=self.embedding,
            start_tokens=start_tokens,
            end_token=end_token,
            softmax_temperature=params['sampling_temperature'],
            seed=params.get('random_seed', None))

        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=cell,
            helper=helper,
            initial_state=initial_state,
            output_layer=output_layer)
        return decoder

    @staticmethod
    def _max_infer_iterations(sequence_length, params):
        if params.get('tgt_max_len_infer', None):
            max_iterations = params['tgt_max_len_infer']
        else:
            decoding_length_factor = 2.0
            max_encoder_length = tf.reduce_max(sequence_length)
            max_iterations = tf.to_int32(tf.round(
                tf.to_float(max_encoder_length) * decoding_length_factor))
        return max_iterations

    def default_config(self):
        config = {
            "num_decoder_layers": 2,
            "num_decoder_residual_layers": 0,
            "num_units": 256,
            "unit_type": "lstm",
            "beam_width": 0,
            "sampling_temperature": 0.0,
            "dropout": 0.5,
            "forget_bias": 1.0,
            "time_major": True,
            "swap_memory": True,
            "unk_id": 0,
            "sos_id": 1,
            "eos_id": 2,
            "infer_batch_size": 32,
        }
        return config
