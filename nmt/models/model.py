import tensorflow as tf
import abc


class Model(abc.ABC):

    def input_fn(self, mode, params):
        raise NotImplementedError()

    def model_fn(self, features, labels, mode, params, config):
        raise NotImplementedError()

    def serving_input_receiver_fn(self):
        receiver_tensors = {
            "source_inputs": tf.placeholder(dtype=tf.string, shape=(None, None)),
            "source_inputs_length": tf.placeholder(dtype=tf.string, shape=(None))
        }
        features = receiver_tensors.copy()
        receiver = tf.estimator.export.ServingInputReceiver(
            features=features,
            receiver_tensors=receiver_tensors)
        return receiver
