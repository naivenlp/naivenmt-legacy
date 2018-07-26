import unittest
import os
from naivenmt.configs import Configs


class TestConfigs(unittest.TestCase):

  def test_configs(self):
    config = Configs(
      os.path.join(os.path.dirname(__file__), "configs.yml")).build()
    self.assertEqual("/tmp/model", config.out_dir)
    self.assertEqual("src", config.src)
    self.assertEqual("tgt", config.tgt)
    self.assertEqual("testdata/train.src", config.src_train_file)


if __name__ == "__main__":
  unittest.main()
