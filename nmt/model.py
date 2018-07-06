import abc


class Model(abc.ABC):

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def eval(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def infer(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def export(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def save(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def restore(self):
        raise NotImplementedError()
