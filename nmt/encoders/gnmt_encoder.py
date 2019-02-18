from nmt.encoders.abstract_encoder import EmbeddingEncoder
import tensorflow as tf
from nmt import rnn_utils


class GNMTEncoder(EmbeddingEncoder):

    def encode(self, inputs, length, mode, params=None):
        default_params = self.default_config()
        if params:
            default_params.update(**params)
        params = default_params

        num_uni_layers = params['num_encoder_layers'] - params['num_bi_layers']

        inputs = self.embedding.embedding(inputs, length, mode, params)
        if params['time_major']:
            inputs = tf.transpose(inputs, perm=[1, 0, 2])

        fw_cell = rnn_utils.build_rnn_cells(
            num_layers=params['num_bi_layers'],
            num_residual_layers=0,
            unit_type=params['unit_type'],
            num_units=params['num_units'],
            dropout=params['dropout'] if mode == tf.estimator.ModeKeys.TRAIN else 0.0,
            forget_bias=params['forget_bias'],
            residual_fn=None)
        bw_cell = rnn_utils.build_rnn_cells(
            num_layers=params['num_bi_layers'],
            num_residual_layers=0,
            unit_type=params['unit_type'],
            num_units=params['num_units'],
            dropout=params['dropout'] if mode == tf.estimator.ModeKeys.TRAIN else 0.0,
            forget_bias=params['forget_bias'],
            residual_fn=None)
        bi_outputs, bi_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=fw_cell,
            cell_bw=bw_cell,
            inputs=inputs,
            sequence_length=length,
            dtype=self.dtype,
            time_major=params['time_major'],
            swap_memory=params['swap_memory'])
        states_bw = bi_states[1]

        uni_cell = rnn_utils.build_rnn_cells(
            num_layers=params['num_encoder_layers'],
            num_residual_layers=params['num_encoder_residual_layers'],
            unit_type=params['unit_type'],
            num_units=params['num_units'],
            dropout=params['dropout'] if mode == tf.estimator.ModeKeys.TRAIN else 0.0,
            forget_bias=params['forget_bias'],
            residual_fn=None)
        uni_outputs, uni_states = tf.nn.dynamic_rnn(
            cell=uni_cell,
            inputs=bi_outputs,
            sequence_length=length,
            dtype=self.dtype,
            time_major=params['time_major'],
            swap_memory=params['swap_memory'])

        outputs, states = uni_outputs, uni_states
        if params['time_major']:
            outputs = tf.transpose(outputs, perm=[1, 0, 2])
        if num_uni_layers == 1:
            states = (states_bw,) + (uni_states,)

        return outputs, states

    def default_config(self):
        config = {
            "num_encoder_layers": 2,
            "num_encoder_residual_layers": 0,
            "num_bi_layers": 1,  # should always be 1.
            "unit_type": "lstm",
            "num_units": 256,
            "dropout": 0.5,
            "forget_bias": 1.0,
            "time_major": True,
            "swap_memory": True
        }
        return config
