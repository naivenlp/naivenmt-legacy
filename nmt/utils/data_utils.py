from nmt.dataset import InferDataset
import tensorflow as tf


def create_train_data(hparams):
    pass


def create_dev_data(hparams):
    pass


def create_test_data(hparams):
    pass


def create_infer_data(hparams, dataset, src_vocab_table):
    infer_dataset = InferDataset(
        src_dataset=dataset,
        src_vocab_table=src_vocab_table,
        batch_size=hparams.batch_size,
        eos=hparams.eos,
        src_max_len=hparams.src_max_len_infer
    )
    return infer_dataset


def merge_files(input_files, output_file, input_files_encoding='UTF-8'):
    if output_file is "":
        raise ValueError("The output file path is empty.")
    if len(input_files) == 0:
        return
    with open(output_file, 'w', encoding='UTF-8') as f0:
        for f in input_files:
            with open(f, 'r', encoding=input_files_encoding) as f1:
                for line in f1.readlines():
                    f0.write(line + '\n')


def merge_train_files(src_files, tgt_files,
                      src_output_file, tgt_output_file,
                      src_files_encoding='UTF-8', tgt_files_encoding='UTF-8'):
    # write src_files to a single src_output_file
    merge_files(src_files, src_output_file, input_files_encoding=src_files_encoding)
    # write tgt_files to a single tgt_output_file
    merge_files(tgt_files, tgt_output_file, input_files_encoding=tgt_files_encoding)


def merge_infer_files(infer_files, output_file, infer_files_encoding='UTF-8'):
    merge_files(infer_files, output_file, input_files_encoding=infer_files_encoding)
