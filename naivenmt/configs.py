import os

import collections
import yaml


class _Configs(
  collections.namedtuple(
    "Configs", ("src", "tgt", "train_prefix", "dev_prefix",
                "test_prefix", "vocab_prefix", "embed_prefix",
                "out_dir", "sos", "eos", "subword_option",
                "check_special_token",
                "src_train_file", "src_dev_file", "src_test_file",
                "tgt_train_file", "tgt_dev_file", "tgt_test_file",
                "src_vocab_file", "tgt_vocab_file",
                "src_embed_file", "tgt_embed_file"))):
  pass


DEFAULT_CONFIGS_FILE = os.path.join(os.path.dirname(__file__), "configs.yml")


class Configs(object):

  def __init__(self, configs_file=DEFAULT_CONFIGS_FILE):
    if not os.path.exists(configs_file):
      raise ValueError("File %s does not exists." % configs_file)
    with open(configs_file) as f:
      self.configs = yaml.load(f)

  def build(self):
    if self.configs["embed_prefix"]:
      src_embed_file = self.configs["embed_prefix"] + "." + self.configs["src"]
      tgt_embed_file = self.configs["embed_prefix"] + "." + self.configs["tgt"]
    else:
      src_embed_file = None
      tgt_embed_file = None
    return _Configs(
      src=self.configs["src"] or "src",
      tgt=self.configs["tgt"],
      train_prefix=self.configs["train_prefix"],
      dev_prefix=self.configs["dev_prefix"],
      test_prefix=self.configs["test_prefix"],
      vocab_prefix=self.configs["vocab_prefix"],
      embed_prefix=self.configs["embed_prefix"],
      out_dir=self.configs["out_dir"],
      sos=self.configs["sos"],
      eos=self.configs["eos"],
      subword_option=self.configs["subword_option"],
      check_special_token=self.configs["check_special_token"],
      src_train_file=self.configs["train_prefix"] + "." + self.configs["src"],
      src_dev_file=self.configs["dev_prefix"] + "." + self.configs["src"],
      src_test_file=self.configs["test_prefix"] + "." + self.configs["src"],
      src_vocab_file=self.configs["vocab_prefix"] + "." + self.configs["src"],
      src_embed_file=src_embed_file,
      tgt_train_file=self.configs["train_prefix"] + "." + self.configs["tgt"],
      tgt_dev_file=self.configs["dev_prefix"] + "." + self.configs["tgt"],
      tgt_test_file=self.configs["test_prefix"] + "." + self.configs["tgt"],
      tgt_vocab_file=self.configs["vocab_prefix"] + "." + self.configs["tgt"],
      tgt_embed_file=tgt_embed_file)
