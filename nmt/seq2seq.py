import tensorflow as tf


class Seq2SeqModel(object):
    def __init__(self,
                 mode,
                 dropout,
                 forget_bias,
                 num_layers,
                 num_units,
                 residual_connection=False,
                 residual_fn=None,
                 device_str=None
                 ):
        self.dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0
        self.forget_bias = forget_bias
        self.num_layers = num_layers
        self.num_units = num_units
        self.residual_connection = residual_connection
        self.residual_fn = residual_fn
        self.device_str = device_str

        self._create_train_model()
        self._create_infer_model()

    def _create_train_model(self):
        pass

    def _create_infer_model(self):
        pass

    def train_and_save_model(self):
        pass

    def load_weights_from_h5(self):
        pass

    def predict(self, input):
        pass
