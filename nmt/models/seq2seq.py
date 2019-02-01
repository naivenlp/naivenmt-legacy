from nmt.models.model import Model


class Seq2Seq(Model):

    def __init__(self,
                 inputter,
                 encoder,
                 decoder):
        self.inputter = inputter
        self.encoder = encoder
        self.decoder = decoder

    def input_fn(self, mode, params):
        return self.inputter.input(mode, params)

    def model_fn(self, features, labels, mode, params, config):
        pass
