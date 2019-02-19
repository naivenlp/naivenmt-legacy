import tensorflow as tf

import abc


class Decoder(abc.ABC):

    def decode(self, outputs, states, labels, model, params):
        raise NotImplementedError()

    def default_config(self):
        raise NotImplementedError()


class EmbeddingDecoder(Decoder):

    def __init__(self, embedding, scope="decoder", dtype=tf.float32):
        self.embedding = embedding
        self.scope = scope
        self.dtype = dtype

    def decode(self, outputs, states, labels, model, params):
        raise NotImplementedError()

    def default_config(self):
        raise NotImplementedError()
