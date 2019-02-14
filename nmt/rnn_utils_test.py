import unittest
from nmt import rnn_utils


class RNNUtilsTest(unittest.TestCase):

    def testBuildRNNCells(self):
        cell0 = rnn_utils.build_rnn_cells(2, 1, "lstm", 16, 0.5, 1.0, None)
        cell0 = rnn_utils.build_rnn_cells(2, 1, "layer_norm_lstm", 16, 0.5, 1.0, None)
        # cell0 = rnn_utils.build_rnn_cells(2, 1, "cudnn_lstm", 16, 0.5, 1.0, None)
        cell0 = rnn_utils.build_rnn_cells(2, 1, "gru", 16, 0.5, 1.0, None)
        # cell0 = rnn_utils.build_rnn_cells(2, 1, "cudnn_gru", 16, 0.5, 1.0, None)
        # cell0 = rnn_utils.build_rnn_cells(2, 1, "lstm_block_cell", 16, 0.5, 1.0, None)
        # cell0 = rnn_utils.build_rnn_cells(2, 1, "lstm_block_fused_cell", 16, 0.5, 1.0, None)
        # cell0 = rnn_utils.build_rnn_cells(2, 1, "gru_block_cell", 16, 0.5, 1.0, None)
        cell0 = rnn_utils.build_rnn_cells(2, 1, "nas", 16, 0.5, 1.0, None)


if __name__ == "__main__":
    unittest.main()
