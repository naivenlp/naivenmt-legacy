import os

import tensorflow as tf
import yaml


class Hparams(object):

  def __init__(self, hparams_file):
    if not os.path.exists(hparams_file):
      raise ValueError("File does not exists: %s" % hparams_file)
    with open(hparams_file) as f:
      self.hparams_config = yaml.load(f)

  def build(self):
    return tf.contrib.training.HParams(
      # Network
      num_units=self.hparams_config["num_units"],
      num_layers=self.hparams_config["num_layers"],
      num_encoder_layers=(self.hparams_config["num_encoder_layers"] or
                          self.hparams_config["num_layers"]),
      num_decoder_layers=(self.hparams_config["num_decoder_layers"] or
                          self.hparams_config["num_layers"]),
      unit_type=self.hparams_config["unit_type"],
      dropout=self.hparams_config["dropout"],
      encoder_type=self.hparams_config["encoder_type"],
      residual=self.hparams_config["residual"],
      time_major=self.hparams_config["time_major"],
      num_embeddings_partitions=self.hparams_config[
        "num_embeddings_partitions"],

      # Attention
      attention=self.hparams_config["attention"],
      attention_architecture=self.hparams_config["attention_architecture"],
      output_attention=self.hparams_config["output_attention"],
      pass_hidden_state=self.hparams_config["pass_hidden_state"],

      # Train
      optimizer=self.hparams_config["optimizer"],
      num_train_steps=self.hparams_config["num_train_steps"],
      batch_size=self.hparams_config["batch_size"],
      init_op=self.hparams_config["init_op"],
      init_weight=self.hparams_config["init_weight"],
      max_gradient_norm=self.hparams_config["max_gradient_norm"],
      learning_rate=self.hparams_config["learning_rate"],
      warmup_steps=self.hparams_config["warmup_steps"],
      warmup_scheme=self.hparams_config["warmup_scheme"],
      decay_scheme=self.hparams_config["decay_scheme"],
      colocate_gradients_with_ops=self.hparams_config[
        "colocate_gradients_with_ops"],

      # Data constraints
      num_buckets=self.hparams_config["num_buckets"],
      max_train=self.hparams_config["max_train"],
      src_max_len=self.hparams_config["src_max_len"],
      tgt_max_len=self.hparams_config["tgt_max_len"],

      # Inference
      src_max_len_infer=self.hparams_config["src_max_len_infer"],
      tgt_max_len_infer=self.hparams_config["tgt_max_len_infer"],
      infer_batch_size=self.hparams_config["infer_batch_size"],
      beam_width=self.hparams_config["beam_width"],
      length_penalty_weight=self.hparams_config["length_penalty_weight"],
      sampling_temperature=self.hparams_config["sampling_temperature"],
      num_translations_per_input=self.hparams_config[
        "num_translations_per_input"],

      # Misc
      forget_bias=self.hparams_config["forget_bias"],
      num_gpus=self.hparams_config["num_gpus"],
      epoch_step=0,
      steps_per_stats=self.hparams_config["steps_per_stats"],
      steps_per_external_eval=self.hparams_config["steps_per_external_eval"],
      share_vocab=self.hparams_config["share_vocab"],
      metrics=self.hparams_config["metrics"].split(","),
      log_device_placement=self.hparams_config["log_device_placement"],
      random_seed=self.hparams_config["random_seed"],
      override_loaded_params=self.hparams_config["override_loaded_params"],
      num_keep_ckpts=self.hparams_config["num_keep_ckpts"],
      avg_ckpts=self.hparams_config["avg_ckpts"],
      num_intra_threads=self.hparams_config["num_intra_threads"],
      num_inter_threads=self.hparams_config["num_inter_threads"])
