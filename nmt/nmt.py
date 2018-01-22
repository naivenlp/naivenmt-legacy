import sys

import tensorflow as tf

from .arguments import arguments

flags = None
default_hparams = None


def run_main(flags, hparams):
    print(flags)
    print(hparams)
    # TODO(luozhouyang) run main logic


def main(unused_argv):
    run_main(flags, default_hparams)


if __name__ == "__main__":
    flags = arguments.get_flags()
    default_hparams = arguments.get_hparams()
    tf.app.run(main=main, argv=[sys.argv[0]] + arguments.get_unparsed())
