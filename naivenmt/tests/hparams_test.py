import argparse
import os

import tensorflow as tf

from naivenmt.configs.arguments import add_arguments
from naivenmt.configs.hparams import Hparams

TEST_DATA_DIR = os.path.abspath(os.path.join(os.pardir, "../", "testdata"))


def add_required_params(flags):
  flags.out_dir = "/tmp/model"
  flags.src = "en"
  flags.tgt = "vi"
  flags.train_prefix = os.path.join(TEST_DATA_DIR, "iwslt15.tst2013.100")
  flags.dev_prefix = os.path.join(TEST_DATA_DIR, "iwslt15.tst2013.100")
  flags.test_prefix = os.path.join(TEST_DATA_DIR, "iwslt15.tst2013.100")
  flags.vocab_prefix = os.path.join(TEST_DATA_DIR, "iwslt15.vocab.100")


class TestHParams(tf.test.TestCase):

  def testHparams(self):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    flags, _ = parser.parse_known_args()
    add_required_params(flags)
    hparams = Hparams(flags).build()

    self.assertEqual("en", hparams.src)
    self.assertEqual(32, hparams.num_units)
    self.assertEqual(2, hparams.num_layers)
    self.assertEqual(2, hparams.num_encoder_layers)
    self.assertEqual(2, hparams.num_decoder_layers)

    # For convenience, print hparams
    print(hparams)


if __name__ == "__main__":
  tf.test.main()
