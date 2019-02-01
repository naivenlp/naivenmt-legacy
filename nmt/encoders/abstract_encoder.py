import abc


class Encoder(abc.ABC):
    """Encoder input sequence."""

    def encode(self, inputs, length, mode, params=None):
        """Encode inputs.

        Args:
            inputs: A tensor, input sequence, shape is [B, T]
            length: A tensor, input sequence's length, shape is [B]
            mode: A python string constant, one of tf.estimator.ModeKeys
            params: A python dict, params

        Returns:
            A output tensor and a states tensor.
        """
        raise NotImplementedError()

    def default_config(self):
        """Default encoding settings."""
        raise NotImplementedError()


class EmbeddingEncoder(Encoder):
    """Embedding input sequence and encode it to produce output and states tensor."""

    def __init__(self, embedding):
        self.embedding = embedding

    def encode(self, inputs, length, mode, params=None):
        raise NotImplementedError()

    def default_config(self):
        raise NotImplementedError()
