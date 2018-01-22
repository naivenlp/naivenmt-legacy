import unittest

from .arguments import arguments


class ArgumentsTest(unittest.TestCase):
    def setUp(self):
        self.flags = arguments.get_flags()
        self.unparsed = arguments.get_unparsed()
        self.hparams = arguments.get_hparams()

    def tearDown(self):
        self.flags = None
        self.unparsed = None
        self.hparams = None

    def testHparams(self):
        self.assertEqual(self.hparams.src, None)
        self.assertEqual(self.hparams.num_units, 32)

    def testFlags(self):
        self.assertEqual(self.hparams.src, None)
        self.assertEqual(self.hparams.num_units, 32)

    def testUnparsed(self):
        self.assertIsNotNone(self.unparsed)


if __name__ == '__main__':
    unittest.main()
