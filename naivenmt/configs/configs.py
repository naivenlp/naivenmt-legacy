import codecs
import os

import tensorflow as tf

UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
UNK_ID = 0


class Configs(object):

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
      unk="<unk>")

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
        unk="<unk>")

    src_embed_file = ""
    tgt_embed_file = ""
    if flags.embed_prefix:
      src_embed_file = flags.embed_prefix + "." + flags.src
      tgt_embed_file = flags.embed_prefix + "." + flags.tgt

    hparams = tf.contrib.training.HParams(
      src=self.flags.src,
      tgt=self.flags.tgt,
      train_prefix=self.flags.train_flags,
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
      inference_ref_file=flags.inference_ref_file)
    # TODO(luozhouyang) Add infer files to inputter

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
