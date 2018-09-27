import tensorflow as tf
from naivenmt.tests import setup_flags
from naivenmt.naivenmt import NaiveNMT
from naivenmt.configs import Hparams


class TestNaiveNMT(tf.test.TestCase):

  def setUp(self):
    self.flags = setup_flags()

  def testNaiveNMTTrain(self):
    nmt = NaiveNMT(Hparams(self.flags).build())
    nmt.train()


if __name__ == "__main__":
  tf.test.main()
