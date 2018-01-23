class AbstractPreprocessor(object):
    def preprocess(self):
        raise NotImplementedError("Not implemented yet.")


class SegmentationPreprocessor(AbstractPreprocessor):
    def segmentation(self):
        raise NotImplementedError("Not implemented yet.")

    def preprocess(self):
        self.segmentation()


class TrainDataPreprocessor(SegmentationPreprocessor):
    def __init__(self, src_file, tgt_file):
        self.src_file = src_file
        self.tgt_file = tgt_file

    def check_files(self):
        pass

    def segmentation(self):
        self.check_files()
        print("Call segmentation method")
        # TODO(luozhouyang): do segmentation


class InferDataPreprocessor(SegmentationPreprocessor):
    def __init__(self, infer_input_file):
        self.infer_input_file = infer_input_file

    def check_infer_input_file(self):
        pass

    def segmentation(self):
        self.check_infer_input_file()
        print("Call segmentation method")
        # TODO(luozhouyang): do segmentation
