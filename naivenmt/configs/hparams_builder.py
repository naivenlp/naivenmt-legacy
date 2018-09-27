import tensorflow as tf
import os
import codecs

__all__ = ["HParamsBuilder"]

UNK = '<unk>'
SOS = '<s>'
EOS = '</s>'
UNK_ID = 0


class HParamsBuilder(object):

  def __init__(self):
    self._hparams = tf.contrib.training.HParams()

  def add_dict(self, **kwargs):
    for k, v in kwargs.items():
      if getattr(self._hparams, k, None) is not None:
        setattr(self._hparams, k, v)
      else:
        self._hparams.add_hparam(k, v)

  def build(self):
    # check vocab
    self._check_vocabs()
    # gen metrics dirs
    self._gen_metrics_dirs()
    return self._hparams

  def _check_vocabs(self):
    src_vocab_file = getattr(self._hparams, 'source_vocab_file', None)
    tgt_vocab_file = getattr(self._hparams, 'target_vocab_file', None)
    assert src_vocab_file
    assert tgt_vocab_file
    check_special_token = getattr(self._hparams, 'check_special_token', True)

    unk = UNK
    sos = getattr(self._hparams, 'sos', SOS)
    eos = getattr(self._hparams, 'eos', EOS)

    def load_vocab(vocab_file):
      vocab = set()
      # print(vocab_file)
      assert tf.gfile.Exists(vocab_file)
      with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
        for word in f:
          word = word.strip()
          # if check_special_token, remove special tokens from vocab first
          if check_special_token and word in [unk, sos, eos]:
            continue
          vocab.add(word)
      vocab = list(sorted(vocab))
      if check_special_token:
        vocab.insert(0, eos)
        vocab.insert(0, sos)
        vocab.insert(0, unk)
      return vocab, len(vocab)

    def gen_new_vocab(vocab, fname):
      with codecs.getwriter("utf-8")(tf.gfile.GFile(fname, "wb+")) as f:
        for word in vocab:
          f.write(word)
          f.write("\n")

    src_vocab, src_vocab_size = load_vocab(src_vocab_file)
    tgt_vocab, tgt_vocab_size = load_vocab(tgt_vocab_file)
    out_dir = getattr(self._hparams, 'out_dir', "/tmp/model")
    # gen new src vocab file in `out_dir`
    new_src_vocab_file = os.path.join(out_dir, os.path.basename(src_vocab_file))
    gen_new_vocab(src_vocab, new_src_vocab_file)
    self._set_hparam('source_vocab_file', new_src_vocab_file)
    self._set_hparam('source_vocab_size', src_vocab_size)

    # gen new tgt vocab file in `out_dir`
    new_tgt_vocab_file = os.path.join(out_dir, os.path.basename(tgt_vocab_file))
    gen_new_vocab(tgt_vocab, new_tgt_vocab_file)
    self._set_hparam('target_vocab_file', new_tgt_vocab_file)
    self._set_hparam('target_vocab_size', tgt_vocab_size)

  def _gen_metrics_dirs(self):
    # gen metrics dirs and add them to hparams
    metrics = getattr(self._hparams, 'metrics', None)
    out_dir = getattr(self._hparams, 'out_dir', "/tmp/model")
    avg_ckpts = getattr(self._hparams, 'avg_ckpts', True)
    if metrics:
      assert isinstance(metrics, list)
      for m in metrics:
        self._hparams.add_hparam("best_" + m, 0)
        best_metric_dir = os.path.join(out_dir, "best_" + m)
        self._hparams.add_hparam("best_" + m + "_dir", best_metric_dir)
        if not tf.gfile.Exists(best_metric_dir):
          tf.gfile.MakeDirs(best_metric_dir)

        if avg_ckpts:
          self._hparams.add_hparam("avg_best_" + m, 0)
          best_avg_metric_dir = os.path.join(out_dir, "avg_best_" + m)
          self._hparams.add_hparam("avg_best_" + m + "_dir",
                                   best_avg_metric_dir)
          if not tf.gfile.Exists(best_avg_metric_dir):
            tf.gfile.MakeDirs(best_avg_metric_dir)

  def _set_hparam(self, name, value):
    try:
      v = getattr(self._hparams, name)
      # tf.logging.INFO("Update hparam %30s: %10s -> %s"
      #                 % (str(name), str(v), str(value)))
      setattr(self._hparams, name, value)
    except AttributeError as e:
      self._hparams.add_hparam(name, value)
      # tf.logging.INFO("Add hparam %30s: %s" % (str(name), str(value)))
