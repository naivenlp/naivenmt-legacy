import tensorflow as tf

from nmt.encoders.abstract_encoder import EmbeddingEncoder
from nmt import rnn_utils


class BiRNNEncoder(EmbeddingEncoder):

    def encode(self, inputs, length, mode, params=None):
        default_params = self.default_config()
        if params:
            default_params.update(**params)
        params = default_params

        inputs = self.embedding.embedding(inputs, length, mode, params)
        if params['time_major']:
            inputs = tf.transpose(inputs, perm=[1, 0, 2])

        num_bi_layers = int(params['num_encoder_layers'] // 2)
        num_bi_residual_layers = int(params['num_encoder_residual_layers'] // 2)
        # TODO(luozhouyang) Add residual_fn
        fw_cell = rnn_utils.build_rnn_cells(
            num_layers=num_bi_layers,
            num_residual_layers=num_bi_residual_layers,
            unit_type=params['unit_type'],
            num_units=params['num_units'],
            dropout=params['dropout'] if mode == tf.estimator.ModeKeys.TRAIN else 0.0,
            forget_bias=params['forget_bias'],
            residual_fn=None)
        bw_cell = rnn_utils.build_rnn_cells(
            num_layers=num_bi_layers,
            num_residual_layers=num_bi_residual_layers,
            unit_type=params['unit_type'],
            num_units=params['num_units'],
            dropout=params['dropout'],
            forget_bias=params['forget_bias'],
            residual_fn=None)

        bi_outputs, bi_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=fw_cell,
            cell_bw=bw_cell,
            inputs=inputs,
            dtype=self.dtype,
            sequence_length=length,
            time_major=params['time_major'],
            swap_memory=params['swap_memory'])
        outputs = tf.concat(bi_outputs, axis=-1)
        if params['time_major']:
            outputs = tf.transpose(outputs, perm=[1, 0, 2])
        # flatten states
        states = []
        for i in range(num_bi_layers):
            states.append(bi_states[0][i])
            states.append(bi_states[1][i])
        return outputs, states

    def default_config(self):
        config = {
            "num_encoder_layers": 2,
            "num_encoder_residual_layers": 0,
            "dropout": 0.5,
            "unit_type": "lstm",
            "time_major": True,
            "swap_memory": True,
            "num_units": 256,
            "forget_bias": 1.0
        }
        return config
