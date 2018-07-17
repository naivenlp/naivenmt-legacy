import tensorflow as tf

from .base_model import BaseModel
from nmt import loss_utils
from nmt import optimize_utils


class Seq2SeqModel(BaseModel):

  def __init__(self,
               name,
               features_inputter,
               labels_inputter,
               encoder,
               decoder):
    super().__init__(name, features_inputter, labels_inputter)
    self.name = name
    self.features_inputter = features_inputter
    self.labels_inputter = labels_inputter
    self.encoder = encoder
    self.decoder = decoder

  def _build(self, features, labels, params, mode, config):
    with tf.variable_scope("encoder"):
      encoder_outputs, encoder_state, encoder_seq_len = self.encoder.encode(
        features,
        configs=config,
        mode=mode)
    labels_vocab_size = self.labels_inputter.vocab_size()
    labels_dtype = self.labels_inputter.dtype

    with tf.variable_scope("decoder") as decoder_scope:
      if labels:
        logits, _, _ = self.decoder.decode(
          labels,
          configs=config,
          mode=mode)
      else:
        logits = None

    if mode != tf.estimator.ModeKeys.TRAIN:
      with tf.variable_scope(decoder_scope,
                             reuse=labels is not None) as decoder_scope:
        beam_width = config.beam_width
        if beam_width <= 1:
          sample_ids, _, sampled_len, log_probs, alignment = (
            self.decoder.dynamic_decode(
              scope=decoder_scope,
              mode=mode,
              vocab_size=labels_vocab_size,
              initial_state=encoder_state,
              memory=encoder_outputs,
              memory_seq_len=encoder_seq_len,
              dtype=labels_dtype,
              return_alignment_history=True))
        else:
          # TODO(luozouyang) Add length_panelty to config
          len_penalty = 0
          sample_ids, _, sampled_len, log_probs, alignment = (
            self.decoder.dynamic_decode_and_search(
              scope=decoder_scope,
              mode=mode,
              vocab_size=labels_vocab_size,
              initial_state=encoder_state,
              beam_width=beam_width,
              length_panelty=len_penalty,
              memory=encoder_outputs,
              memory_seq_len=encoder_seq_len,
              dtype=labels_dtype,
              return_alignment_history=True))

      target_vocab_reverse = self.labels_inputter.vocab_reverse()
      target_tokens = target_vocab_reverse.lookup(tf.cast(sample_ids, tf.int64))
      if config.replace_unknown_target:
        source_tokens = features["tokens"]
        if beam_width > 1:
          source_tokens = tf.contrib.seq2seq.tile_batch(source_tokens,
                                                        multiplier=beam_width)
        target_tokens = self._replace_unknown_target(source_tokens,
                                                     target_tokens)
      prediction = {
        "tokens": target_tokens,
        "length": sampled_len,
        "log_probs": log_probs
      }
      if alignment:
        prediction["alignment"] = alignment
    else:
      prediction = None

    return logits, prediction

  def _initializer(self, params):
    init_value = params.init_value
    if init_value:
      return tf.random_uniform_initializer(
        -init_value, init_value, dtype=self.dtype)
    return None

  def _optimize(self, loss, params):
    return optimize_utils.optimize(loss, params)

  def _compute_metrics(self, features, labels, predictions):
    pass

  def _compute_loss(self, features, labels, outputs, params, mode):
    return loss_utils.cross_entropy_sequence_loss(
      logits=outputs,
      labels=labels["ids_out"],
      sequence_length=self.labels_inputter.get_length(labels),
      smoothing=params.smoothing,
      average_in_time=params.averaget_in_time,
      mode=mode)

  def _replace_unknown_target(self, source_tokens, target_tokens):
    raise NotImplementedError()
