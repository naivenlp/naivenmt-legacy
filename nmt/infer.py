from .utils import data_utils
from .utils import model_utils


def infer(hparams):
    infer_data = data_utils.create_infer_data(hparams)
    infer_model = model_utils.create_infer_model(hparams)
