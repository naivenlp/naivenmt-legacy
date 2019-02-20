import tensorflow as tf

from nmt.decoders.basic_decoder import BasicRNNDecoder


class BeamRNNDecoder(BasicRNNDecoder):

    def _build_decoder(self, cell, initial_state, params):
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

    def default_config(self):
        config = super(BeamRNNDecoder, self).default_config()
        config.update({
            "beam_width": 5,
            "sampling_temperature": 0.0,
            "length_penalty_weight": 1.0
        })
        return config
