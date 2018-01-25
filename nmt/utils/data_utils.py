from nmt.dataset import InferDataset, TrainDataset


def create_train_data(hparams,
                      src_dataset,
                      tgt_dataset,
                      src_vocab_table,
                      tgt_vocab_table):
    train_dataset = TrainDataset(
        src_dataset=src_dataset,
        tgt_dataset=tgt_dataset,
        src_vocab_table=src_vocab_table,
        tgt_vocab_table=tgt_vocab_table,
        batch_size=hparams.batch_size,
        sos=hparams.sos,
        eos=hparams.eos,
        source_reverse=hparams.source_reverse,
        random_seed=hparams.random_seed,
        num_buckets=hparams.num_buckets,
        src_max_len=hparams.src_max_len,
        tgt_max_len=hparams.tgt_max_len,
        num_parallel_calls=hparams.num_parallel_calls,
        output_buffer_size=hparams.output_buffer_size,
        skip_count=hparams.skip_count,
        num_shards=hparams.num_shards,
        shard_index=hparams.shard_index
    )
    return train_dataset


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
