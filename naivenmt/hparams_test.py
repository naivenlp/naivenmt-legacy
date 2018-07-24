import unittest
from .hparams import Hparams


class TestHParams(unittest.TestCase):

  def test_hparams(self):
    hparams = Hparams("naivenmt/hparams.yml").build()
    self.assertEqual("/tmp/model", hparams.out_dir)
    self.assertEqual(None, hparams.random_seed)
    self.assertEqual(None, hparams.decay_scheme)
    self.assertEqual(True, hparams.avg_ckpts)
    self.assertEqual(5.0, hparams.max_gradient_norm)


if __name__ == "__main__":
  unittest.main()
