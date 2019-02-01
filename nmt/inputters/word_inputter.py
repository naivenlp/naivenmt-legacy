import tensorflow as tf

from nmt import collections
from nmt.inputters.inputter import Inputter


def _split_src_tgt_fn(x):
    return x.split('@')


def _split_src_word_fn(src):
    raise NotImplementedError()


def _split_tgt_word_fn(tgt):
    raise NotImplementedError()


class WordInputter(Inputter):

    def input(self, mode, params):
        default_params = self.default_config()
        if not params:
            params = default_params
        else:
            params = default_params.update(**params)

        if mode == tf.estimator.ModeKeys.TRAIN:
            return self._build_train_input(params)
        elif mode == tf.estimator.ModeKeys.EVAL:
            return self._build_eval_input(params)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            return self._build_predict_input(params)

    def _build_train_input(self, params):
        input_files = params['train_files'].split(",")
        return self._build_input(input_files, params)

    def _build_eval_input(self, params):
        input_files = params['eval_files'].split(",")
        return self._build_input(input_files, params)

    def _build_input(self, input_files, params):
        dataset = tf.data.Dataset.from_tensor_slices(input_files)
        dataset = dataset.flat_map(lambda f: tf.data.TextLineDataset(f).skip(params['skip_count']))

        dataset = dataset.repeat(params['repeat']).prefetch(params['buff_size'])
        dataset = dataset.shuffle(
            buffer_size=params['buff_size'],
            seed=params['random_seed'],
            reshuffle_each_iteration=params['reshuffle_each_iteration']
        ).prefetch(params['buff_size'])

        dataset = dataset.map(
            lambda x: _split_src_tgt_fn(x),
            num_parallel_calls=params['num_parallel_calls']
        ).prefetch(params['buff_size'])

        dataset = dataset.map(
            lambda src, tgt: (_split_src_word_fn(src), _split_tgt_word_fn(tgt)),
            num_parallel_calls=params['num_parallel_calls']
        ).prefetch(params['buff_size'])

        dataset = dataset.filter(
            lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0)
        ).prefetch(params['buff_size'])

        dataset = dataset.map(
            lambda src, tgt: (src[:params['src_max_len']], tgt),
            num_parallel_calls=params['num_parallel_calls']
        ).prefetch(params['buff_size'])

        dataset = dataset.map(
            lambda src, tgt: (src, tgt[:params['tgt_max_len']]),
            num_parallel_calls=params['num_parallel_calls']
        ).prefetch(params['buff_size'])

        dataset = dataset.map(
            lambda src, tgt: (
                src,
                tf.concat(([params['sos']], tgt), 0),
                tf.concat((tgt, [params['eos']]), 0)),
            num_parallel_calls=params['num_parallel_calls']
        ).prefetch(params['buff_size'])

        dataset = dataset.map(
            lambda src, tgt_in, tgt_out: (src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_out)),
            num_parallel_calls=params['num_parallel_calls']
        ).prefetch(params['buff_size'])

        dataset = dataset.padded_batch(
            batch_size=params['batch_size'],
            padded_shapes=(
                tf.TensorShape([None]),
                tf.TensorShape([None]),
                tf.TensorShape([None]),
                tf.TensorShape([]),
                tf.TensorShape([])),
            padding_values=(
                params['eos'],
                params['sos'],
                params['eos'],
                0,
                0)
        ).prefetch(params['buff_size'])

        iterator = dataset.make_initializable_iterator()
        collections.add_to_collection(collections.ITERATOR, iterator)

        src, tgt_in, tgt_out, src_len, tgt_len = iterator.get_next()
        features = {
            "source_inputs": src,
            "source_inputs_length": src_len,
        }
        labels = {
            "target_in": tgt_in,
            "target_out": tgt_out,
            "target_inputs_length": tgt_len
        }
        return features, labels

    def _build_predict_input(self, params):
        input_files = params['predict_files'].split(",")
        dataset = tf.data.Dataset.from_tensor_slices(input_files)
        dataset = dataset.flat_map(lambda f: tf.data.TextLineDataset(f).skip(params['skip_count']))

        dataset = dataset.map(
            lambda x: _split_src_word_fn(x),
            num_parallel_calls=params['num_parallel_calls']
        ).prefetch(params['buff_size'])

        dataset = dataset.map(
            lambda x: x[:params['predict_max_len']],
            num_parallel_calls=params['num_parallel_calls']
        ).prefetch(params['buff_size'])

        dataset = dataset.map(
            lambda x: (x, tf.size(x)),
            num_parallel_calls=params['num_parallel_calls']
        ).prefetch(params['buff_size'])

        dataset = dataset.padded_batch(
            batch_size=params['predict_batch_size'],
            padded_shapes=(
                tf.TensorShape([None]),
                tf.TensorShape([])),
            padding_values=(
                tf.constant(params['sos'], dtype=tf.string),
                0)
        ).prefetch(params['buff_size'])

        iterator = dataset.make_initializable_iterator()
        collections.add_to_collection(collections.ITERATOR, iterator)

        inputs, length = iterator.get_next()
        features = {"source_inputs": inputs, "source_inputs_length": length}
        return features, None

    def default_config(self):
        params = {
            "skip_count": 0,
            "num_parallel_calls": 4,
            "buff_size": 8192,
            "sos": "<s>",
            "eos": "</s>",
            "predict_src_max_len": None,
            "src_max_len": None,
            "tgt_max_len": None,
            "random_seed": 1000000,
            "reshuffle_each_iteration": True,
            "repeat": None
        }
        return params
