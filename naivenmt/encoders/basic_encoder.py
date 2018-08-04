import tensorflow as tf

from naivenmt.encoders.abstract_encoder import AbstractEncoder


class BasicEncoder(AbstractEncoder):
  """Basic encoder."""

  def __init__(self,
               params,
               embedding,
               scope=None,
               dtype=None,
               single_cell_fn=None):
    """Init encoder.

    Args:
      params: hparams
      embedding: embedding, an instance of ``naivenmt.embeddings.Embedding``
      scope: variables scope
      dtype: variables dtype
      single_cell_fn: a func to process single rnn cell
    """
    super().__init__(params=params,
                     embedding=embedding,
                     scope=scope,
                     dtype=dtype)
    self.single_cell_fn = single_cell_fn
    if not self.single_cell_fn:
      self.single_cell_fn = self._single_cell_fn

  def _build_encoder_cell(self,
                          mode, num_layers, num_residual_layers, base_gpu=0):
    return self._create_rnn_cells(
      num_layers=num_layers,
      num_residual_layers=num_residual_layers,
      mode=mode,
      base_gpu=base_gpu,
      single_cell_fn=self.single_cell_fn)

  def _create_rnn_cells(self,
                        num_layers,
                        num_residual_layers,
                        mode,
                        single_cell_fn,
                        residual_fn=None,
                        base_gpu=0):
    cells = self._create_rnn_cell_list(
      num_layers=num_layers,
      num_residual_layers=num_residual_layers,
      mode=mode,
      single_cell_fn=single_cell_fn,
      residual_fn=residual_fn,
      base_gpu=base_gpu)
    if len(cells) == 1:
      return cells[0]
    else:
      return tf.contrib.rnn.MultiRNNCell(cells)

  def _create_rnn_cell_list(self,
                            num_layers,
                            num_residual_layers,
                            mode,
                            single_cell_fn,
                            residual_fn=None,
                            base_gpu=0):
    cells = []
    for i in range(num_layers):
      residual_conn = (i >= num_layers - num_residual_layers)
      device = self._get_device_str(i + base_gpu, self.num_gpus)
      cell = single_cell_fn(
        unit_type=self.unit_type,
        num_units=self.num_units,
        forget_bias=self.forget_bias,
        dropout=self.dropout,
        mode=mode,
        device=device,
        residual_conn=residual_conn,
        residual_fn=residual_fn)
      cells.append(cell)
    return cells

  @staticmethod
  def _single_cell_fn(unit_type,
                      num_units,
                      forget_bias,
                      dropout,
                      mode,
                      device=None,
                      residual_conn=False,
                      residual_fn=None):
    """Create a single rnn cell.

    Args:
      unit_type: unit type of cell
      num_units: number of cells
      forget_bias: forget bias for lstm cells
      dropout: dropout
      mode: mode
      device: device that holds this cell
      residual_conn: If use residual connection or not
      residual_fn: process residual connections if use residual connections

    Returns:
      A single rnn cell
    """

    dropout = dropout if mode == tf.estimator.ModeKeys.TRAIN else 0.0

    if unit_type == "lstm":
      single_cell = tf.contrib.rnn.BasicLSTMCell(
        num_units, forget_bias=forget_bias)
    elif unit_type == "gru":
      single_cell = tf.contrib.rnn.GRUCell(num_units)
    elif unit_type == "layer_norm_lstm":
      single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
        num_units, forget_bias=forget_bias, layer_norm=True)
    elif unit_type == "nas":
      single_cell = tf.contrib.rnn.NASCell(num_units)
    else:
      raise ValueError("Invalid unit type: %s" % unit_type)

    if dropout > 0.0:
      single_cell = tf.contrib.rnn.DropoutWrapper(
        cell=single_cell, input_keep_prob=(1.0 - dropout))
    if residual_conn:
      single_cell = tf.contrib.rnn.ResidualWrapper(
        single_cell, residual_fn=residual_fn)

    if device:
      single_cell = tf.contrib.rnn.DeviceWrapper(
        single_cell, device)
    return single_cell

  @staticmethod
  def _get_device_str(device_id, num_gpus):
    if num_gpus == 0:
      return "/cpu:0"
    device_str = "/gpu:%d" % (device_id % num_gpus)
    return device_str
