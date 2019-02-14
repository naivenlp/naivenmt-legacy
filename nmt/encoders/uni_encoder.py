import tensorflow as tf

from nmt import rnn_utils
from nmt.encoders.abstract_encoder import EmbeddingEncoder


class UniRNNEncoder(EmbeddingEncoder):

    def encode(self, inputs, length, mode, params=None):
        default_params = self.default_config()
        params = default_params if not params else default_params.update(**params)

        inputs = self.embedding.embedding(inputs, length, params)
        if params['time_major']:
            inputs = tf.transpose(inputs, perm=[1, 0, 2])

        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE) as scope:
            cell = rnn_utils.build_rnn_cells(
                num_layers=params['num_encoder_layers'],
                num_residual_layers=params['num_encoder_residual_layers'],
                unit_type=params['unit_type'],
                num_units=params['num_units'],
                dropout=params['dropout'] if mode == tf.estimator.ModeKeys.TRAIN else 0.0,
                forget_bias=params['forget_bias'],
                residual_fn=None
            )
            outputs, states = tf.nn.dynamic_rnn(
                cell=cell,
                inputs=inputs,
                dtype=self.dtype,
                sequence_length=length,
                time_major=params['time_major'],
                swap_memory=params['swap_memory'])

        if params['time_major']:
            outputs = tf.transpose(outputs, perm=[1, 0, 2])

        return outputs, states

    def default_config(self):
        params = {
            "num_encoder_layers": 2,
            "num_encoder_residual_layers": 0,
            "dropout": 0.5,
            "unit_type": "lstm",
            "time_major": True,
            "swap_memory": True,
            "num_units": 256,
            "forget_bias": 1.0
        }
        return params
