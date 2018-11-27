import os

import tensorflow as tf

from naivenmt.configs import HParamsBuilder


class TestHparamsBuilder(tf.test.TestCase):

  def testHParamsBuilder(self):
    config_dir = os.path.abspath(
      os.path.join(os.path.dirname(__file__), "../configs"))
    print(config_dir)
    builder = HParamsBuilder(os.path.join(config_dir, "example_hparams.json"))
    hparams = builder.build()
    builder.print_configs()


if __name__ == "__main__":
  tf.test.main()
