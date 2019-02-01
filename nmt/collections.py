import tensorflow as tf

ITERATOR = "dataset_iterator"


def add_to_collection(key, value):
    tf.add_to_collection(key, value)


def get_from_collection(key):
    return tf.get_collection(key)


def add_dict_to_collection(name, _dict):
    for k, v in _dict.items():
        tf.add_to_collection(name + "_key", k)
        tf.add_to_collection(name + "_value", v)


def get_dict_from_collection(name):
    keys = tf.get_collection(name + "_key")
    values = tf.get_collection(name + "_value")
    return dict(zip(keys, values))
