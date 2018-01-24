import sys

import tensorflow as tf

from . import infer
from . import train
from .arguments import arguments

flags = None
default_hparams = None


def main(_):
    if flags.inference_input_file is not None:
        infer.infer(default_hparams)
    else:
        train.train(default_hparams)


if __name__ == "__main__":
    flags = arguments.get_flags()
    default_hparams = arguments.get_hparams()
    tf.app.run(main=main, argv=[sys.argv[0]] + arguments.get_unparsed())
