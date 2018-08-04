# Copyright 2018 luozhouyang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import codecs
import os

import tensorflow as tf

UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
UNK_ID = 0


class Hparams(object):

  def __init__(self, flags):
    self.flags = flags

  def build(self):
    flags = self.flags

    if flags.subword_option and flags.subword_option not in ["spm", "bpe"]:
      raise ValueError("subword option must be either spm, or bpe")

    if not tf.gfile.Exists(flags.out_dir):
      tf.gfile.MakeDirs(flags.out_dir)

    assert flags.train_prefix
    assert flags.dev_prefix
    assert flags.test_prefix

    src_train_file = flags.train_prefix + "." + flags.src
    tgt_train_file = flags.train_prefix + "." + flags.tgt
    src_dev_file = flags.dev_prefix + "." + flags.src
    tgt_dev_file = flags.dev_prefix + "." + flags.tgt
    src_test_file = flags.test_prefix + "." + flags.src
    tgt_test_file = flags.dev_prefix + "." + flags.tgt

    if not flags.vocab_prefix:
      raise ValueError("vocab_prefix must be provided.")
    src_vocab_file = flags.vocab_prefix + "." + flags.src
    tgt_vocab_file = flags.vocab_prefix + "." + flags.tgt
    src_vocab_file, src_vocab_size = self._check_vocab(
      vocab_file=src_vocab_file,
      out_dir=flags.out_dir,
      check_special_token=flags.check_special_token,
      sos=flags.sos,
      eos=flags.eos,
      unk=UNK)

    if flags.share_vocab:
      tgt_vocab_file = src_vocab_file
      tgt_vocab_size = src_vocab_size
    else:
      tgt_vocab_file, tgt_vocab_size = self._check_vocab(
        vocab_file=tgt_vocab_file,
        out_dir=flags.out_dir,
        check_special_token=flags.check_special_token,
        sos=flags.sos,
        eos=flags.eos,
        unk=UNK)

    src_embed_file = ""
    tgt_embed_file = ""
    if flags.embed_prefix:
      src_embed_file = flags.embed_prefix + "." + flags.src
      tgt_embed_file = flags.embed_prefix + "." + flags.tgt

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
      # Files
      src=self.flags.src,
      tgt=self.flags.tgt,
      train_prefix=self.flags.train_prefix,
      dev_prefix=self.flags.dev_prefix,
      test_prefix=self.flags.test_prefix,
      vocab_prefix=self.flags.vocab_prefix,
      embed_prefix=self.flags.embed_prefix,
      out_dir=self.flags.out_dir,
      sos=self.flags.sos,
      eos=self.flags.eos,
      source_train_file=src_train_file,
      target_train_file=tgt_train_file,
      source_dev_file=src_dev_file,
      target_dev_file=tgt_dev_file,
      source_test_file=src_test_file,
      target_test_file=tgt_test_file,
      subword_option=self.flags.subword_option,
      check_special_token=self.flags.check_special_token,
      source_vocab_file=src_vocab_file,
      source_vocab_size=src_vocab_size,
      target_vocab_file=tgt_vocab_file,
      target_vocab_size=tgt_vocab_size,
      source_embedding_file=src_embed_file,
      target_embedding_file=tgt_embed_file,
      ckpt=flags.ckpt,  # inference ckpt
      inference_input_file=flags.inference_input_file,
      inference_output_file=flags.inference_output_file,
      inference_list=flags.inference_list,
      inference_ref_file=flags.inference_ref_file,

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
      override_loaded_params=self.flags.override_loaded_hparams,
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

  def _check_vocab(self,
                   vocab_file,
                   out_dir,
                   check_special_token=True,
                   sos=SOS,
                   eos=EOS,
                   unk=UNK):
    if tf.gfile.Exists(vocab_file):
      print("# Vocab file %s exists" % vocab_file)
      vocab, vocab_size = self._load_vocab(vocab_file)
      if check_special_token:
        # Verify if the vocab starts with unk, sos, eos
        # If not, prepend those tokens & generate a new vocab file
        assert len(vocab) >= 3
        if vocab[0] != unk or vocab[1] != sos or vocab[2] != eos:
          print("The first 3 vocab words [%s, %s, %s]"
                " are not [%s, %s, %s]" %
                (vocab[0], vocab[1], vocab[2], unk, sos, eos))
          vocab = [unk, sos, eos] + vocab
          vocab_size += 3
          new_vocab_file = os.path.join(out_dir, os.path.basename(vocab_file))
          with codecs.getwriter("utf-8")(
                  tf.gfile.GFile(new_vocab_file, "wb")) as f:
            for word in vocab:
              f.write("%s\n" % word)
          vocab_file = new_vocab_file
    else:
      raise ValueError("vocab_file '%s' does not exist." % vocab_file)

    vocab_size = len(vocab)
    return vocab_file, vocab_size

  @staticmethod
  def _load_vocab(vocab_file):
    vocab = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
      for word in f:
        vocab.append(word.strip())
    return vocab, len(vocab)
