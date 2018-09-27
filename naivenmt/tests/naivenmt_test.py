import tensorflow as tf
import yaml

from naivenmt.configs import HParamsBuilder
from naivenmt.naivenmt import NaiveNMT
from naivenmt.tests import common_test_utils as utils


class TestNaiveNMT(tf.test.TestCase):

  def testNaiveNMTTrain(self):
    builder = HParamsBuilder()
    with open(utils.get_file_path('configs', 'default_hparams.yml'),
              encoding='utf8') as f:
      config = yaml.load(f)
    builder.add_dict(**config)
    utils.set_test_files_hparams(builder)
    hparams = builder.build()

    print(hparams.out_dir)
    print(hparams.num_train_steps)
    nmt = NaiveNMT(hparams)
    nmt.train()


if __name__ == "__main__":
  tf.test.main()
