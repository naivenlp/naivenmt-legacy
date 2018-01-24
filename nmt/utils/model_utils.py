import tensorflow as tf
from tensorflow.contrib.keras.api.keras.layers import Input, Embedding, Dense
from tensorflow.contrib.keras.api.keras.models import Sequential


def create_train_model(hparams):
    model = Sequential()
    # TODO(luozhouyang): add layers to model
    # for test
    model.add(Input(shape=(None, None), batch_size=32, dtype=tf.int32))
    model.add(Embedding(output_dim=128, input_dim=1000))
    model.add(Dense(units=32, activation='softmax'))
    return model


def create_infer_model(hparams):
    model = Sequential()
    # TODO(luozhouyang): add layers to model

    return model
