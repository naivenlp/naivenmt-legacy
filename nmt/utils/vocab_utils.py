from tensorflow.python.ops import lookup_ops

SOS = "<s>"
EOS = "</s>"
UNK = "<unk>"
UNK_ID = 0


def create_vocab_table(src_vocab_file, tgt_vocab_file, share_vocab):
    src_vocab_table = lookup_ops.index_table_from_file(src_vocab_file, default_value=UNK_ID)
    if share_vocab:
        tgt_vocab_table = src_vocab_table
    else:
        tgt_vocab_table = lookup_ops.index_table_from_file(tgt_vocab_file, default_value=UNK_ID)
    return src_vocab_table, tgt_vocab_table
