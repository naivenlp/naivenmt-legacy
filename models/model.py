import keras.backend as K
import tensorflow as tf
from keras import Model
from keras.initializers import RandomUniform, glorot_normal, glorot_uniform
from keras.layers import Input, LSTM, Embedding, GRU, Dense, RNN
from keras.optimizers import SGD, Adam

from utils import vocab_utils


class StandardModel(object):

    def __init__(self, config, mode):
        self.config = config
        self._init_vocabs()
        self.model = self._build_model(mode)

    def train(self):
        optimizer = self._get_optimizer()
        loss = self._get_loss()
        metrics = self.config.configs['train']['metrics']
        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=metrics, )

        train_batch_size = self.config.configs['train']['batch_size']
        # TODO(luozhouyang) add `num_epochs` arg (tf/nmt doesn't have it)
        epochs = 10
        x_train, y_train = self._get_train_data_set()
        x_validate, y_validate = self._get_validate_data_set()
        x_test, y_test = self._get_test_data_set()

        for _ in range(epochs):
            # TODO(luozhouyang) set `initial_epoch` not 0 to recover training
            # TODO(luozhouyang) add callbacks to custom training
            self.model.fit(x=x_train, y=y_train,
                           batch_size=train_batch_size,
                           epochs=1,
                           verbose=2,
                           callbacks=None,
                           validation_data=(x_validate, y_validate),
                           initial_epoch=0)
        score = self.model.evaluate(x=x_test, y=y_test, batch_size=128)
        print("Evaluate score: %s" % score)

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
        # TODO(luozhouyang) support residual network
        # TODO(luozhouyang) support bi-direction network and decide the size

        encoder_states = [state_h, state_c]
        decoder_input = Input(dtype=tf.string,
                              batch_shape=(train_batch_size, None),
                              name='decoder_input')
        x = Embedding(input_dim=self.tgt_vocab_len + 1,
                      output_dim=embedding_size,
                      mask_zero=True,
                      input_length=self.max_tgt_len)(decoder_input)

        # is this correct to create multi-layer RNN?
        cells = self._build_rnn_cells(mode, num_decoder_layers, return_seq=True)
        x = RNN(cell=cells)(x, initial_state=encoder_states)
        # TODO(luozhouyang) support residual network
        # TODO(luozhouyang) support bi-direction network and decide the size

        decoder_output = Dense(units=self.tgt_vocab_len + 1,
                               activation='softmax')(x)

        return Model([encoder_input, decoder_input], decoder_output)

    def _build_rnn_cells(self, mode, num_layers, return_seq=False):
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
        # TODO(luozhouyang) Decide which initializer to set
        if 'lstm' == unit_type:
            # already set forget bias in default
            cell = LSTM(units=num_units, return_sequences=return_seq,
                        dropout=dropout, recurrent_dropout=dropout)
        elif 'gru' == unit_type:
            cell = GRU(units=num_units, return_sequences=return_seq,
                       dropout=dropout, recurrent_dropout=dropout)

        # TODO(luozhouyang) Add other unit types
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

    def _get_optimizer(self):
        opt = self.config.configs['model']['optimizer']
        lr = self.config.configs['model']['learning_rate']
        decay = self.config.configs['model']['decay_factor']
        if 'sgd' == opt:
            # TODO(luozhouyang) decay learning rate
            return SGD(lr=lr, )
        elif 'adam' == opt:
            return Adam()
        else:
            raise ValueError("%s optimizer is not supported yet.")

    def _get_loss(self):
        def _loss(y_true, y_pred):
            return K.sparse_categorical_crossentropy(y_true, y_pred, True)

        return _loss

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

    # TODO(luozhouyang) fetch data
    def _get_train_data_set(self):
        return '', ''

    def _get_validate_data_set(self):
        return '', ''

    def _get_test_data_set(self):
        return '', ''
