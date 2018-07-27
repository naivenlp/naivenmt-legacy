from naivenmt.encoders.abstract_encoder import AbstractEncoder
from naivenmt import utils


class BasicEncoder(AbstractEncoder):

  def _build_encoder_cell(self, mode, params, num_layers, num_residual_layers,
                          base_gpu=0):
    return utils.create_rnn_cells(
      unit_type=params.unit_type,
      num_units=params.num_units,
      num_layers=num_layers,
      num_residual_layers=num_residual_layers,
      forget_bias=params.forget_bias,
      dropout=params.dropout,
      num_gpus=params.num_gpus,
      mode=mode,
      base_gpu=base_gpu,
      single_cell_fn=self._single_cell_fn)

  def _create_single_cell_fn(self):
    return utils.single_cell_fn
