from nmt.decoders.abstract_decoder import EmbeddingDecoder
import tensorflow as tf
from nmt import rnn_utils


class BasicRNNDecoder(EmbeddingDecoder):

    def decode(self, outputs, states, labels, mode, params):
        default_params = self.default_config()
        if params:
            default_params.update(**params)
        params = default_params

        with tf.variable_scope(self.scope, dtype=self.dtype, reuse=tf.AUTO_REUSE) as scope:
            cell, initial_state = rnn_utils.build_decoder_rnn_cells(
                states=states,
                num_layers=params['num_decoder_layers'],
                num_residual_layers=params['num_decoder_residual_layers'],
                num_units=params['num_units'],
                unit_type=params['unit_type'],
                beam_width=params['beam_width'],
                dropout=params['dropout'],
                forget_bias=params['forget_bias'],
                residual_fn=None)

            if mode != tf.estimator.ModeKeys.PREDICT:
                helper = tf.contrib.seq2seq.TrainingHelper(
                    inputs=labels['tgt_in'],
                    sequence_length=labels['tgt_len'],
                    time_major=params['time_major'])
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=cell,
                    helper=helper,
                    initial_state=initial_state)
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder,
                    output_time_major=False,
                    swap_memory=params['swap_memory'])
                sample_id = outputs.sample_id
                logits = tf.layers.dense(inputs=outputs.rnn_output, units=params['target_vocab_size'], use_bias=False)
                return logits, sample_id, final_context_state

            return self._decode(cell, initial_state, params)

    def _decode(self, cell, initial_state, params):
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
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder,
            maximum_iterations=params['max_iteration'],
            output_time_major=False,
            swap_memory=params['swap_memory'])
        logits = outputs.rnn_output
        sample_id = outputs.sample_id
        return logits, sample_id, final_context_state

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
