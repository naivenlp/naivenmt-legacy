import abc


class Inputter(abc.ABC):
    """Sequence inputter."""

    def input(self, mode, params):
        """Build input pipeline for TRAIN, EVAL and PREDICT mode.

        Args:
            mode: A python string, one of tf.estimator.ModeKeys
            params: A python dict, params
        """
        raise NotImplementedError()

    def default_config(self):
        """Default settings."""
        raise NotImplementedError()
