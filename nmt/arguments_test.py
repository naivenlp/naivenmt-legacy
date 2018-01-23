import unittest

from .arguments import arguments


class ArgumentsTest(unittest.TestCase):
    def testGetHparams(self):
        hparams = arguments.get_hparams()
        self.assertEqual(hparams.src, None)
        self.assertEqual(hparams.num_units, 32)

    def testGetFlags(self):
        flags = arguments.get_hparams()
        self.assertEqual(flags.src, None)
        self.assertEqual(flags.num_units, 32)

    def testGetUnparsed(self):
        unparsed = arguments.get_unparsed()
        self.assertIsNotNone(unparsed)

    def testCreateHparams(self):
        hparams = arguments._create_hparams()
        self.assertEqual(hparams.src, None)
        self.assertEqual(hparams.num_units, 32)


if __name__ == '__main__':
    unittest.main()
