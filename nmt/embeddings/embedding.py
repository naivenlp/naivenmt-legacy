import abc


class Embedding(abc.ABC):
    """Embedding interface."""

    def embedding(self, inputs, length, params=None):
        """Do embedding.

        Args:
            inputs: A tensor, input sequence, shape is [B, T]
            length: A tensor, input's length, shape is [B]
            params: A python dict, optional, params

        Returns:
            A tensor, represent input's embedding, shape is [B, T, D]
        """
        raise NotImplementedError()

    def default_config(self):
        """Default params settings.

        Returns:
            A python dict, default params
        """
        raise NotImplementedError()
