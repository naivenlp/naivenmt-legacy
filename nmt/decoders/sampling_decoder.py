import tensorflow as tf

from nmt.decoders.basic_decoder import BasicRNNDecoder


class SamplingRNNDecoder(BasicRNNDecoder):

    def _build_decoder(self, cell, initial_state, params):
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

    def default_config(self):
        config = super(SamplingRNNDecoder, self).default_config()
        config.update({
            "beam_width": 0,
            "sampling_temperature": 1.0,
            "length_penalty_weight": 0.0
        })
        return config
