import os


def get_test_data_dir():
    cur = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(cur, '../testdata')
    return os.path.abspath(data_dir)


def get_test_data(filename):
    return os.path.join(get_test_data_dir(), filename)
