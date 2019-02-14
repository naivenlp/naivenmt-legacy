import tensorflow as tf


# TODO(luozhouyang): ensure MultiRNNCell accepts block RNN cells
def build_rnn_cells(
        num_layers,
        num_residual_layers,
        unit_type,
        num_units,
        dropout,
        forget_bias,
        residual_fn=None):
    # cudnn cells and block cell are not supported yet.
    # if unit_type == "cudnn_lstm":
    #     cells = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers, num_units)
    #     return cells
    #
    # if unit_type == "cudnn_gru":
    #     cells = tf.contrib.cudnn_rnn.CudnnGRU(num_layers, num_units)
    #     return cells

    cells = []
    for i in range(num_layers):
        residual = (i >= num_layers - num_residual_layers)
        if unit_type == "lstm":
            cell = tf.nn.rnn_cell.LSTMCell(num_units=num_units, forget_bias=forget_bias)
        elif unit_type == "layer_norm_lstm":
            cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units, forget_bias, layer_norm=True)
        # elif unit_type == "lstm_block_fused_cell":
        #     cell = tf.contrib.rnn.LSTMBlockFusedCell(num_units=num_units, forget_bias=forget_bias)
        # elif unit_type == "lstm_block_cell":
        #     cell = tf.contrib.rnn.LSTMBlockCell(num_units=num_units, forget_bias=forget_bias)
        # elif unit_type == "gru_block_cell":
        #     cell = tf.contrib.rnn.GRUBlockCellV2(num_units=num_units)
        elif unit_type == "gru":
            cell = tf.nn.rnn_cell.GRUCell(num_units=num_units)
        elif unit_type == "nas":
            cell = tf.contrib.rnn.NASCell(num_units=num_units)
        else:
            raise ValueError("Invalid unit_type: %s" % unit_type)

        if dropout > 0.0:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, 1.0 - dropout)
        if residual and residual_fn:
            cell = tf.nn.rnn_cell.ResidualWrapper(cell, residual_fn)

        cells.append(cell)

    return tf.nn.rnn_cell.MultiRNNCell(cells)
