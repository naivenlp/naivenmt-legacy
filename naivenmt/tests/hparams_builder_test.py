import os
import json
import tensorflow as tf

from naivenmt.configs import HParamsBuilder


class TestHparamsBuilder(tf.test.TestCase):

  def testHParamsBuilder(self):
    config_dir = os.path.abspath(
      os.path.join(os.path.dirname(__file__), "../configs"))
    print(config_dir)
    hparams_file = os.path.join(config_dir, "example_hparams.json")
    with open(hparams_file, mode="rt", encoding="utf8") as f:
      builder = HParamsBuilder(json.load(f))
    hparams = builder.build()
    builder.print_configs()


if __name__ == "__main__":
  tf.test.main()
