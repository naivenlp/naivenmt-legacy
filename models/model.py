import tensorflow as tf
from keras import Model
from keras.layers import Input, LSTM, Embedding, GRU, Dense, RNN
from keras.initializers import RandomUniform, glorot_normal, glorot_uniform
from utils import vocab_utils


class StandardModel(object):

    def __init__(self, config, mode):
        self.config = config
        self._init_vocabs()
        self.model = self._build_model(mode)

    def _build_model(self, mode):
        train_batch_size = self.config.configs['train']['batch_size']
        embedding_size = self.config.configs['model']['embedding_size']
        num_encoder_layers = self.config.configs['model']['num_encoder_layers']
        num_decoder_layers = self.config.configs['model']['num_decoder_layers']

        # one of ['uni' 'bi' 'gnmt'] in tensorflow/nmt
        encoder_type = self.config.configs['model']['encoder_type']
        # LSTM set forget bias in default
        forget_bias = self.config.configs['model']['forget_bias']

        encoder_input = Input(dtype=tf.string,
                              batch_shape=(train_batch_size, None),
                              name='encoder_input')
        x = Embedding(input_dim=self.src_vocab_len + 1,
                      output_dim=embedding_size,
                      mask_zero=True,
                      input_length=self.max_src_len)(encoder_input)
        cells = self._build_rnn_cells(mode, num_encoder_layers)
        x, state_h, state_c = RNN(cell=cells)(x)
        # TODO(allen.luo) support residual network
        # TODO(allen.luo) support bi-direction network and decide the size

        encoder_states = [state_h, state_c]
        decoder_input = Input(dtype=tf.string,
                              batch_shape=(train_batch_size, None),
                              name='decoder_input')
        x = Embedding(input_dim=self.tgt_vocab_len + 1,
                      output_dim=embedding_size,
                      mask_zero=True,
                      input_length=self.max_tgt_len)(decoder_input)

        # is this correct to create multi-layer RNN?
        cells = self._build_rnn_cells(mode, num_decoder_layers)
        x = RNN(cell=cells)(x, initial_state=encoder_states)
        # TODO(allen.luo) support residual network
        # TODO(allen.luo) support bi-direction network and decide the size

        decoder_output = Dense(units=self.tgt_vocab_len + 1,
                               activation='softmax')(x)

        return Model([encoder_input, decoder_input], decoder_output)

    def _build_rnn_cells(self, mode, num_layers):
        num_layers = num_layers if num_layers > 0 else 1
        unit_type = self.config.configs['model']['unit_type']
        num_units = self.config.configs['model']['num_units']
        if mode == tf.contrib.learn.ModeKeys.LEARN:
            dropout = self.config.configs['model']['dropout']
        else:
            dropout = 0.0

        initializer = self._get_initializer()
        # keras has many initializer to be set,
        # but tf/nmt seems only set once.
        # TODO(allen.luo) Decide which initializer to set
        if 'lstm' == unit_type:
            # already set forget bias in default
            cell = LSTM(units=num_units, return_sequences=True,
                        dropout=dropout, recurrent_dropout=dropout)
        elif 'gru' == unit_type:
            cell = GRU(units=num_units, return_sequences=True,
                       dropout=dropout, recurrent_dropout=dropout)

        # TODO(allen.luo) Add other unit types
        else:
            raise ValueError("%s is not a supported unit type." % unit_type)

        cells = []
        for _ in range(num_layers):
            cells.append(cell)
        return cells

    def _get_initializer(self):
        """Get initializer.

            Tensorflow/nmt currently support
            ['uniform', 'glorot_normal', 'glorot_uniform'] init ops.

            We can add more ops in the future
        """
        initializer = None
        init_op = self.config.configs['model']['init_op']
        random_seed = self.config.configs['model']['random_seed']
        init_weight = self.config.configs['model']['init_weight']
        if init_op == 'uniform':
            initializer = RandomUniform(-init_weight, init_weight, random_seed)
        elif init_op == 'glorot_normal':
            initializer = glorot_normal(random_seed)
        elif init_op == 'glorot_uniform':
            initializer = glorot_uniform
        else:
            raise ValueError("%s init_op is not supported yet." % init_op)
        return initializer

    def _init_vocabs(self):
        src_vocab_file = self.config.configs['data']['src_vocab']
        src_vocab = vocab_utils.parse_vocab(src_vocab_file)
        share_vocab = self.config.configs['train']['share_vocab']
        if share_vocab:
            tgt_vocab = src_vocab
        else:
            tgt_vocab_file = self.config.configs['data']['tgt_vocab']
            tgt_vocab = vocab_utils.parse_vocab(tgt_vocab_file)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_vocab_len = len(src_vocab)
        self.tgt_vocab_len = len(tgt_vocab)
        self.max_src_len = self.config.configs['train']['max_src_len']
        self.max_tgt_len = self.config.configs['train']['max_tgt_len']
