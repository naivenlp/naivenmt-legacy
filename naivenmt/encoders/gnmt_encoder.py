import tensorflow as tf

from naivenmt.encoders.basic_encoder import BasicEncoder


class GNMTEncoder(BasicEncoder):
  """GNMT encoder."""

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
                     dtype=dtype,
                     single_cell_fn=single_cell_fn)

  def encode(self, mode, features):
    if self.encoder_type in ["uni", "bi"]:
      raise ValueError("uni or bi encoder type only support BasicEncoder.")

    if self.encoder_type != "gnmt":
      raise ValueError("Invalid encoder type: %s" % self.encoder_type)

    num_bi_layers = 1
    num_uni_layers = self.num_encoder_layers - num_bi_layers
    sequence_length = features.source_sequence_length

    with tf.variable_scope("encoder", dtype=self.dtype) as scope:
      bi_encoder_outputs, bi_encoder_state = self._build_bidirectional_rnn(
        mode=mode,
        inputs=self.embedding.encoder_embedding_input(features.source_ids),
        sequence_length=sequence_length,
        num_bi_layers=num_bi_layers,
        num_bi_residual_layers=0)

      uni_cell = self._create_rnn_cells(
        num_layers=num_uni_layers,
        num_residual_layers=self.num_encoder_residual_layers,
        mode=mode,
        base_gpu=1,
        single_cell_fn=self.single_cell_fn)

      encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
        uni_cell,
        bi_encoder_outputs,
        dtype=self.dtype,
        sequence_length=sequence_length,
        time_major=self.time_major)

      encoder_state = (bi_encoder_state[1],) + (
        (encoder_state,) if num_uni_layers == 1 else encoder_state)

    return encoder_outputs, encoder_state
