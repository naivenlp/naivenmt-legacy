import os

import tensorflow as tf
import yaml
import json

from naivenmt.configs import HParamsBuilder
from naivenmt.tests import common_test_utils as utils


class TestHparamsBuilder(tf.test.TestCase):

  def testBuildFromYamlFile(self):
    builder = HParamsBuilder()
    config = utils.get_file_path('configs', 'default_hparams.yml')
    print(config)

    with open(config, mode="rt", encoding='utf8') as f:
      configs = yaml.load(f)
      for k, v in configs.items():
        print("%30s -> %s" % (k, v))

    builder.add_dict(**configs)
    hparams = builder.build()
    for k, v in configs.items():
      self.assertEqual(v, getattr(hparams, k, None))

  def testBuildFromMultiFiles(self):

    json_config = utils.get_file_path('configs', 'wmt16.json')
    yaml_config = utils.get_file_path('configs', 'default_hparams.yml')

    print(json_config)
    print(yaml_config)

    with open(yaml_config, mode="rt", encoding='utf8') as f:
      yaml_configs = yaml.load(f)
      for k, v in yaml_configs.items():
        print("%30s -> %s" % (k, v))

    print("========================================================")

    with open(json_config, mode="rt", encoding="utf8") as f:
      json_configs = json.load(f)
      for k, v in yaml_configs.items():
        print("%30s -> %s" % (k, v))

    builder = HParamsBuilder()
    builder.add_dict(**json_configs)
    # yaml config will override json config
    builder.add_dict(**yaml_configs)
    hparams = builder.build()
    for k, v in yaml_configs.items():
      self.assertEqual(v, getattr(hparams, k, None))

    builder = HParamsBuilder()
    builder.add_dict(**yaml_configs)
    # json config will override yaml config
    builder.add_dict(**json_configs)
    hparams = builder.build()
    for k, v in json_configs.items():
      self.assertEqual(v, getattr(hparams, k, None))

    builder = HParamsBuilder()
    configs = {**json_configs, **yaml_configs}
    builder.add_dict(**configs)
    hparams = builder.build()
    for k, v in configs.items():
      self.assertEqual(v, getattr(hparams, k, None))


if __name__ == "__main__":
  tf.test.main()
