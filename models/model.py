import tensorflow as tf
from keras import Model
from keras.layers import Input, LSTM, Embedding, GRU, Dense

from utils import vocab_utils


class StandardModel(object):

    def __init__(self, config, mode):
        self.config = config
        self.model = self.build_model(mode)

    def build_model(self, mode):
        src_vocab_file = self.config.configs['data']['src_vocab']
        src_vocab = vocab_utils.parse_vocab(src_vocab_file)
        share_vocab = self.config.configs['train']['share_vocab']
        if share_vocab:
            tgt_vocab = src_vocab
        else:
            tgt_vocab_file = self.config.configs['data']['tgt_vocab']
            tgt_vocab = vocab_utils.parse_vocab(tgt_vocab_file)
        src_vocab_len = len(src_vocab)
        tgt_vocab_len = len(tgt_vocab)
        max_src_len = self.config.configs['train']['max_src_len']
        max_tgt_len = self.config.configs['train']['max_tgt_len']

        train_batch_size = self.config.configs['train']['batch_size']
        embedding_size = self.config.configs['model']['embedding_size']
        num_encoder_layers = self.config.configs['model']['num_encoder_layers']
        num_decoder_layers = self.config.configs['model']['num_decoder_layers']
        encoder_type = self.config.configs['model']['encoder_type']
        forget_bias = self.config.configs['model']['forget_bias']

        encoder_input = Input(shape=(None,), dtype=tf.string,
                              batch_shape=(train_batch_size, None),
                              name='encoder_input')
        x = Embedding(input_dim=src_vocab_len + 1, output_dim=embedding_size,
                      mask_zero=True, input_length=max_src_len)(encoder_input)
        x, state_h, state_c = self._build_rnn(mode, num_encoder_layers)(x)
        # TODO(allen.luo) support residual network

        # TODO(allen.luo) support bi-direction network and decide the size

        encoder_states = [state_h, state_c]

        decoder_input = Input(shape=(None,), dtype=tf.string,
                              batch_shape=(train_batch_size, None),
                              name='decoder_input')
        x = Embedding(input_dim=tgt_vocab_len + 1, output_dim=embedding_size,
                      mask_zero=True, input_length=max_tgt_len)(decoder_input)
        x = self._build_rnn(
            mode, num_decoder_layers)(x, initial_state=encoder_states)
        decoder_output = Dense(units=tgt_vocab_len, activation='softmax')(x)

        return Model([encoder_input, decoder_input], decoder_output)

    def _build_rnn(self, mode, num_layers):
        # TODO(allen.luo) Add multi layer RNN
        unit_type = self.config.configs['model']['unit_type']
        num_units = self.config.configs['model']['num_units']
        if mode == tf.contrib.learn.ModeKeys.LEARN:
            dropout = self.config.configs['model']['dropout']
        else:
            dropout = 0.0
        # TODO(allen.luo) set forget bias
        if 'lstm' == unit_type:
            return LSTM(units=num_units, return_sequences=True,
                        dropout=dropout, recurrent_dropout=dropout)
        elif 'gru' == unit_type:
            return GRU(units=num_units, return_sequences=True,
                       dropout=dropout, recurrent_dropout=dropout)
        # TODO(allen.luo) Add other unit types
        else:
            raise ValueError("%s is not a supported unit type." % unit_type)
