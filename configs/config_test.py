import json
import sys
import unittest

from configs.config import Config


class TestConfig(unittest.TestCase):

  def test_config(self):
    config = Config()
    print(json.dump(config.configs, sys.stdout, indent=4))


if __name__ == "__main__":
  unittest.main()
