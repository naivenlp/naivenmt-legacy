class AbstractTranslationModel(object):

    def input(self):
        raise NotImplementedError("Not implemented yet.")

    def embedding(self):
        raise NotImplementedError("Not implemented yet.")

    def encode(self):
        raise NotImplementedError("Not implemented yet.")

    def decode(self):
        raise NotImplementedError("Not implemented yet.")

    def output(self):
        raise NotImplementedError("Not implemented yet.")

    def translate(self):
        self.input()
        self.embedding()
        self.encode()
        self.decode()
        self.output()


class TranslationModel(AbstractTranslationModel):
    def input(self):
        pass

    def embedding(self):
        pass

    def encode(self):
        pass

    def decode(self):
        pass

    def output(self):
        pass


class AddressCorrectionModel(TranslationModel):

    def do_segmentation(self):
        pass

    def translate(self):
        self.do_segmentation()
        super().translate()


class ChineseAddressCorrectionModel(AddressCorrectionModel):
    def do_segmentation(self):
        pass
