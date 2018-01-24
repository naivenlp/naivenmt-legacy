from tensorflow.contrib.keras.api.keras import optimizers

from .arguments import arguments
from .utils import callback_utils
from .utils import model_utils


def train(hparams):
    train_model = model_utils.create_train_model(hparams)

    callbacks = callback_utils.create_callbacks(hparams)

    optimizer = None
    if hparams.optimizer == "sgd":
        optimizer = optimizers.SGD(lr=arguments.learning_rate(),
                                   decay=0.5)
    elif hparams.optimizer == "adam":
        optimizer = optimizers.Adam(lr=0.001, decay=0.5)

    train_model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=arguments.metrics(),
        sample_weight_mode=None,
        weighted_metrics=None,
        target_tensors=None
    )

    train_model.fit(batch_size=arguments.batch_size(),
                    epochs=arguments.num_train_steps())
