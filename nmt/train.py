from .utils import callback_utils
from .utils import data_utils
from .utils import model_utils


def train(hparams):
    train_data = data_utils.create_train_data(hparams)
    dev_data = data_utils.create_dev_data(hparams)
    test_data = data_utils.create_test_data(hparams)

    train_model = model_utils.create_train_model(hparams)

    callbacks = callback_utils.create_callbacks(hparams)
