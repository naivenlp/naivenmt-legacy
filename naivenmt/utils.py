import tensorflow as tf


def get_device_str(device_id, num_gpus):
  if num_gpus == 0:
    return "/cpu:0"
  device_str = "/gpu:%d" % (device_id % num_gpus)
  return device_str


def create_rnn_cells(unit_type,
                     num_units,
                     num_layers,
                     num_residual_layers,
                     forget_bias,
                     dropout,
                     mode,
                     num_gpus,
                     single_cell_fn,
                     residual_fn=None,
                     base_gpu=0):
  cells = create_rnn_cell_list(unit_type=unit_type,
                               num_units=num_units,
                               num_layers=num_layers,
                               num_residual_layers=num_residual_layers,
                               forget_bias=forget_bias,
                               dropout=dropout,
                               mode=mode,
                               num_gpus=num_gpus,
                               single_cell_fn=single_cell_fn,
                               residual_fn=residual_fn,
                               base_gpu=base_gpu)
  if len(cells) == 1:
    return cells[0]
  else:
    return tf.contrib.rnn.MultiRNNCell(cells)


def create_rnn_cell_list(unit_type,
                         num_units,
                         num_layers,
                         num_residual_layers,
                         forget_bias,
                         dropout,
                         mode,
                         num_gpus,
                         single_cell_fn,
                         residual_fn=None,
                         base_gpu=0):
  cells = []
  for i in range(num_layers):
    residual_conn = (i >= num_layers - num_residual_layers)
    device = get_device_str(i + base_gpu, num_gpus)
    cell = single_cell_fn(unit_type, num_units, forget_bias, dropout,
                          mode, device, residual_conn, residual_fn)
    cells.append(cell)
  return cells


def single_cell_fn(unit_type,
                   num_units,
                   forget_bias,
                   dropout,
                   mode,
                   device=None,
                   residual_conn=False,
                   residual_fn=None):
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
