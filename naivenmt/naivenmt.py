import abc
import argparse
from naivenmt.configs import Configs
from naivenmt.hparams import Hparams


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

  def __init__(self, configs_file, hparams_file):
    self.configs = Configs(configs_file=configs_file)
    self.hparams = Hparams(hparams_file=hparams_file).build()

  def train(self):
    pass

  def eval(self):
    pass

  def predict(self):
    pass

  def export(self):
    pass


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--mode", type=str,
                      choices=["train", "eval", "predict", "export"],
                      default="train",
                      help="Which operation you want to do.")
  parser.add_argument("--configs_file", type=str, required=True,
                      help="Configs file.")
  parser.add_argument("--hparams_file", type=str, required=True,
                      help="Hparams file.")

  args, _ = parser.parse_known_args()
  naive_nmt = NaiveNMT(args.configs_file, args.hparams_file)
  if args.mode == "train":
    naive_nmt.train()
  elif args.mode == "eval":
    naive_nmt.eval()
  elif args.mode == "predict":
    naive_nmt.predict()
  elif args.mode == "export":
    naive_nmt.export()
  else:
    raise ValueError("Unknown mode: %s" % args.mode)


if __name__ == "__main__":
  main()
