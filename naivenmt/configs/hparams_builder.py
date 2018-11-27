import os

import tensorflow as tf

__all__ = ["HParamsBuilder"]


class HParamsBuilder(object):
  """Build hparams."""

  def __init__(self, dict_config=None):
    """Constructor.

    Args:
      dict_config: A dict of params
    """
    self._hparams = tf.contrib.training.HParams()
    self.configs = self._get_default_configs()
    if dict_config is not None:
      self.configs.update(dict_config)

  def build(self):
    self._check_data_files()
    self._check_vocab_files()
    self._gen_metrics_dirs()
    self._check_model_params()

    # convert dict to hparams
    for k, v in self.configs.items():
      try:
        getattr(self._hparams, k)
        self._hparams.set_hparam(k, v)
      except Exception as e:
        self._hparams.add_hparam(k, v)
    return self._hparams

  def _check_model_params(self):
    """Check neural network's parameters."""
    enc_type = self.configs['encoder_type']
    if enc_type not in ['uni', 'bi', 'gnmt']:
      raise ValueError("encoder type must be one of ['uni', 'bi', 'gnmt'].")

    attn = self.configs['attention']
    if attn not in ['', 'luong', 'scaled_luong', 'bahdanau', 'normed_bahdanau']:
      raise ValueError(
        "attention must be one of "
        "['', 'luong', 'scaled_luong', 'bahdanau', 'normed_bahdanau'].")

    num_enc_layers = self.configs['num_encoder_layers']
    num_dec_layers = self.configs['num_decoder_layers']
    if not num_enc_layers or not num_dec_layers:
      raise ValueError(
        "num_encoder_layers and num_decoder_layers must be greater than 0.")
    if num_enc_layers != num_dec_layers:
      self.configs['pass_hidden_state'] = False

    if enc_type == "bi" and num_enc_layers % 2 != 0:
      raise ValueError(
        "num_encoder_layers must be even when encoder_type is %s." % enc_type)

    attn_arch = self.configs.get('attention_architecture', None)
    if attn_arch in ["gnmt"] and num_enc_layers < 2:
      raise ValueError("For gnmt attention architecture, "
                       "num_encoder_layers: %d should be >= 2." %
                       num_enc_layers)

    infer_mode = self.configs['infer_mode']
    beam_width = self.configs.get("beam_width", 0)
    if infer_mode == "beam_search" and beam_width <= 0:
      raise ValueError("beam_width must be > 0 if infer_mode is `beam_search`.")

    sample_temp = self.configs.get("sampling_temperature", 0.0)
    if infer_mode == "sample" and sample_temp <= 0.0:
      raise ValueError(
        "sampling_temperature must greater than 0.0 using sample decode.")

    subword_option = self.configs['subword_option']
    if subword_option not in ['', 'bpe', 'spm']:
      raise ValueError("subword_option must be one of ['','bpe','spm']")

    num_enc_residual_layers = 0
    num_dec_residual_layers = 0
    if self.configs['residual']:
      if num_enc_layers > 1:
        num_enc_residual_layers = num_enc_layers - 1
      if num_dec_layers > 1:
        num_dec_residual_layers = num_dec_layers - 1

      if enc_type == "gnmt":
        num_enc_residual_layers = num_enc_layers - 2
        if num_enc_layers == num_dec_layers:
          num_dec_residual_layers = num_enc_residual_layers

    self.configs['num_encoder_residual_layers'] = num_enc_residual_layers
    self.configs['num_decoder_residual_layers'] = num_dec_residual_layers

  def _gen_metrics_dirs(self):
    """Generate metrics dirs."""
    metrics = self.configs['metrics'].split(",")
    for m in metrics:
      if not m or not m.strip():
        continue
      best_m_dir = os.path.join(self.configs['out_dir'], "best_" + m)
      self.configs['best_' + str(m) + "_dir"] = best_m_dir
      if not tf.gfile.Exists(best_m_dir):
        tf.gfile.MakeDirs(best_m_dir)

      if not self.configs['avg_ckpts']:
        continue
      avg_best_m_dir = os.path.join(self.configs['out_dir'], "avg_best_" + m)
      self.configs['avg_best_' + str(m) + "_dir"] = avg_best_m_dir
      if not tf.gfile.Exists(avg_best_m_dir):
        tf.gfile.MakeDirs(avg_best_m_dir)

  def _check_data_files(self):
    """Check data file."""
    # check out dir
    if not tf.gfile.Exists(self.configs['out_dir']):
      try:
        tf.gfile.MakeDirs(self.configs['out_dir'])
      except tf.errors.OpError as e:
        raise Exception("out dir: %s create failed." % self.configs['out_dir'])

    src_train_file = self.configs['train_prefix'] + "." + self.configs['src']
    src_dev_file = self.configs['dev_prefix'] + "." + self.configs['src']
    src_test_file = self.configs['test_prefix'] + "." + self.configs['src']
    self._check_files_exist([src_train_file, src_dev_file, src_test_file])
    self.configs['source_train_file'] = src_train_file
    self.configs['source_dev_file'] = src_dev_file
    self.configs['source_test_file'] = src_test_file

    tgt_train_file = self.configs['train_prefix'] + "." + self.configs['tgt']
    tgt_dev_file = self.configs['dev_prefix'] + "." + self.configs['tgt']
    tgt_test_file = self.configs['test_prefix'] + "." + self.configs['tgt']
    self._check_files_exist([tgt_train_file, tgt_dev_file, tgt_test_file])
    self.configs['target_train_file'] = tgt_train_file
    self.configs['target_dev_file'] = tgt_dev_file
    self.configs['target_test_file'] = tgt_test_file

    if self.configs['embed_prefix']:
      self.configs['source_embed_file'] = (
          self.configs['embed_prefix'] + "." + self.configs['src'])
      self.configs['target_embed_file'] = (
          self.configs['embed_prefix'] + "." + self.configs['tgt'])
    else:
      self.configs['source_embed_file'] = None
      self.configs['target_embed_file'] = None

  def _check_vocab_files(self):
    """Check src and tgt vocab files, adding special tokens to it."""
    src_vocab = self.configs['vocab_prefix'] + "." + self.configs['src']
    src_vocab_size, src_vocab_file = self._check_vocab_file(
      src_vocab,
      [self.configs['unk'], self.configs['sos'], self.configs['eos']])
    self.configs['source_vocab_file'] = src_vocab_file
    self.configs['source_vocab_size'] = src_vocab_size

    tgt_vocab = self.configs['vocab_prefix'] + "." + self.configs['tgt']
    tgt_vocab_size, tgt_vocab_file = self._check_vocab_file(
      tgt_vocab,
      [self.configs['unk'], self.configs['sos'], self.configs['eos']])
    self.configs['target_vocab_file'] = tgt_vocab_file
    self.configs['target_vocab_size'] = tgt_vocab_size

  def _check_vocab_file(self, vocab_file, special_tokens):
    """Check vocab files, adding special tokens to it.

    Args:
      vocab_file: The file path of original vocab file
      special_tokens: A list of special tokens

    Returns:
      The vocab size of new vocab file.
    """
    if not os.path.exists(vocab_file):
      raise FileNotFoundError("vocab file %s not found!" % vocab_file)
    vocabs = set()
    with open(vocab_file, mode="rt", encoding="utf8", buffering=8192) as f:
      for vocab in f:
        vocab = vocab.strip("\n").strip()
        if not vocab:
          continue
        if vocab in special_tokens:
          continue
        vocabs.add(vocab)
    vocabs = sorted(vocabs)
    for token in reversed(special_tokens):
      vocabs.insert(0, token)
    filename = str(vocab_file).split(os.sep)[-1]
    assert filename is not None
    new_vocab_file = os.path.join(self.configs['out_dir'], filename)
    with open(new_vocab_file, mode="wt", encoding="utf8", buffering=8192) as f:
      for v in vocabs:
        f.write(v + "\n")
    return len(vocabs), new_vocab_file

  def print_configs(self):
    for k, v in self.configs.items():
      print("%30s : %s" % (k, v))

  @staticmethod
  def _get_default_configs():
    """Generate default params."""
    testdata_dir = os.path.abspath(
      os.path.join(os.path.dirname(__file__), "../../testdata"))
    return {
      "src": "en",
      "tgt": "vi",
      "unk": "<unk>",
      "sos": "<s>",
      "eos": "</s>",
      "out_dir": os.path.join(testdata_dir, "tmp", "model"),
      "train_prefix": os.path.join(testdata_dir, "iwslt15.tst2013.100"),
      "dev_prefix": os.path.join(testdata_dir, "iwslt15.tst2013.100"),
      "test_prefix": os.path.join(testdata_dir, "iwslt15.tst2013.100"),
      "vocab_prefix": os.path.join(testdata_dir, "iwslt15.vocab.100"),
      "batch_size": 64,
      "embed_prefix": None,
      "metrics": "bleu",  # comma separated string
      "avg_ckpts": False,
      "encoder_type": "uni",
      "residual": True,
      "num_encoder_layers": 2,
      "num_decoder_layers": 2,
      "infer_mode": "greedy",
      "attention": "",
      "attention_architecture": "standard",
      "output_attention": True,
      "pass_hidden_state": True,
      "optimizer": "sgd",
      "learning_rate": 1.0,
      "subword_option": ""
    }

  @staticmethod
  def _check_files_exist(files):
    for file in files:
      if not os.path.exists(file):
        raise FileNotFoundError("File %s not found!" % file)
