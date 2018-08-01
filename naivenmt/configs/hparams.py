import tensorflow as tf
import os


class Hparams(object):

  def __init__(self, flags):
    self.flags = flags

  def build(self):
    flags = self.flags

    assert flags.num_layers
    if not flags.num_encoder_layers:
      flags.num_encoder_layers = flags.num_layers
    if not flags.num_decoder_layers:
      flags.num_decoder_layers = flags.num_layers

    if flags.num_encoder_layers != flags.num_decoder_layers:
      flags.pass_hidden_state = False
      print("num_encoder_layers %d is different from num_decoder_layers %d. "
            "set pass_hidden_state=False." % (
              flags.num_encoder_layers,
              flags.num_decoder_layers))

    if flags.encoder_type == "bi" and flags.num_encoder_layers % 2 != 0:
      raise ValueError("For bi, num_encoder_layers should be even.")

    if flags.attention_architecture in [
      "gnmt"] and flags.num_encoder_layers < 2:
      raise ValueError("For gnmt attention architecture, "
                       "num_encoder_layer should >= 2.")

    num_encoder_residual_layers = 0
    num_decoder_residual_layers = 0
    if flags.residual:
      if flags.num_encoder_layers > 1:
        num_encoder_residual_layers = flags.num_encoder_layers - 1
      if flags.num_decoder_layers > 1:
        num_decoder_residual_layers = flags.num_decoder_layers - 1

      if flags.encoder_type == "gnmt":
        num_encoder_residual_layers = flags.num_encoder_layers - 2
        if flags.num_encoder_layers == flags.num_decoder_layers:
          num_decoder_residual_layers = num_encoder_residual_layers

    source_embedding_size = flags.num_units
    target_embedding_size = flags.num_units

    if not tf.gfile.Exists(flags.out_dir):
      tf.gfile.MakeDirs(flags.out_dir)

    metrics = flags.metrics.split(",")

    hparams = tf.contrib.training.HParams(
      # Network
      num_units=self.flags.num_units,
      num_layers=self.flags.num_layers,
      num_encoder_layers=self.flags.num_encoder_layers,
      num_decoder_layers=self.flags.num_decoder_layers,
      num_encoder_residual_layers=num_encoder_residual_layers,
      num_decoder_residual_layers=num_decoder_residual_layers,
      unit_type=self.flags.unit_type,
      dropout=self.flags.dropout,
      encoder_type=self.flags.encoder_type,
      residual=self.flags.residual,
      time_major=self.flags.time_major,
      source_embedding_size=source_embedding_size,
      target_embedding_size=target_embedding_size,
      num_embeddings_partitions=self.flags.num_embeddings_partitions,

      # Attention
      attention=self.flags.attention,
      attention_architecture=self.flags.attention_architecture,
      output_attention=self.flags.output_attention,
      pass_hidden_state=self.flags.pass_hidden_state,

      # Train
      optimizer=self.flags.optimizer,
      num_train_steps=self.flags.num_train_steps,
      batch_size=self.flags.batch_size,
      init_op=self.flags.init_op,
      init_weight=self.flags.init_weight,
      max_gradient_norm=self.flags.max_gradient_norm,
      learning_rate=self.flags.learning_rate,
      warmup_steps=self.flags.warmup_steps,
      warmup_scheme=self.flags.warmup_scheme,
      decay_scheme=self.flags.decay_scheme,
      colocate_gradients_with_ops=self.flags.colocate_gradients_with_ops,

      # Data constraints
      num_buckets=self.flags.num_buckets,
      max_train=self.flags.max_train,
      src_max_len=self.flags.src_max_len,
      tgt_max_len=self.flags.tgt_max_len,

      # Inference
      src_max_len_infer=self.flags.src_max_len_infer,
      tgt_max_len_infer=self.flags.tgt_max_len_infer,
      infer_batch_size=self.flags.infer_batch_size,
      beam_width=self.flags.beam_width,
      length_penalty_weight=self.flags.length_penalty_weight,
      sampling_temperature=self.flags.sampling_temperature,
      num_translations_per_input=self.flags.num_translations_per_input,

      # Misc
      forget_bias=self.flags.forget_bias,
      num_gpus=self.flags.num_gpus,
      epoch_step=0,
      steps_per_stats=self.flags.steps_per_stats,
      steps_per_external_eval=self.flags.steps_per_external_eval,
      share_vocab=self.flags.share_vocab,
      metrics=metrics,
      log_device_placement=self.flags.log_device_placement,
      random_seed=self.flags.random_seed,
      override_loaded_params=self.flags.override_loaded_params,
      num_keep_ckpts=self.flags.num_keep_ckpts,
      avg_ckpts=self.flags.avg_ckpts,
      num_intra_threads=self.flags.num_intra_threads,
      num_inter_threads=self.flags.num_inter_threads)

    for metric in metrics:
      hparams.add_hparam("best_" + metric, 0)  # larger is better
      best_metric_dir = os.path.join(flags.out_dir, "best_" + metric)
      hparams.add_hparam("best_" + metric + "_dir", best_metric_dir)
      tf.gfile.MakeDirs(best_metric_dir)

      if hparams.avg_ckpts:
        hparams.add_hparam("avg_best_" + metric, 0)  # larger is better
        best_metric_dir = os.path.join(flags.out_dir, "avg_best_" + metric)
        hparams.add_hparam("avg_best_" + metric + "_dir", best_metric_dir)
        tf.gfile.MakeDirs(best_metric_dir)

    return hparams
