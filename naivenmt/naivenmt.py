import abc
import argparse
import sys

import tensorflow as tf

from naivenmt.configs import add_arguments
from naivenmt.configs import Configs
from naivenmt.configs import Hparams


class NaiveNMTInterface(abc.ABC):

  @abc.abstractmethod
  def train(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def eval(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def predict(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def export(self):
    raise NotImplementedError()


class NaiveNMT(NaiveNMTInterface):

  def __init__(self, configs, hparams):
    self.configs = configs
    self.hparams = hparams

  def train(self):
    pass

  def eval(self):
    pass

  def predict(self):
    pass

  def export(self):
    pass


def main(unused_args):
  configs = Configs(FLAGS).build()
  hparams = Hparams(FLAGS).build()
  naive_nmt = NaiveNMT(configs, hparams)
  if FLAGS.mode == "train":
    naive_nmt.train()
  elif FLAGS.mode == "eval":
    naive_nmt.eval()
  elif FLAGS.mode == "predict":
    naive_nmt.predict()
  elif FLAGS.mode == "export":
    naive_nmt.export()
  else:
    raise ValueError("Unknown mode: %s" % FLAGS.mode)


FLAGS = None
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--mode", type=str,
                      choices=["train", "eval", "predict", "export"],
                      default="train",
                      help="Run mode.")
  add_arguments(parser)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
