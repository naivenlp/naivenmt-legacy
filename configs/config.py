import os

import yaml

from .contants import Constants
from .data_config import DataConfigInterface
from .evaluation_config import EvaluationConfigInterface
from .inference_config import InferenceConfigInterface
from .params_config import ParamsConfigInterface
from .training_config import TrainingConfigInterface


# TODO(luozhouyang) Add default value, and test
class Config(DataConfigInterface,
             ParamsConfigInterface,
             TrainingConfigInterface,
             EvaluationConfigInterface,
             InferenceConfigInterface):

  def __init__(self, config_file=os.path.join(os.path.dirname(__file__),
                                              "config.yml")):
    self.config_file = config_file
    with open(config_file) as f:
      self.configs = yaml.load(f)

  @property
  def optimizer(self):
    return self.configs[Constants.GROUP_PARAMS][Constants.KEY_OPTIMIZER]

  @property
  def learning_rate(self):
    return self.configs[Constants.GROUP_PARAMS][Constants.KEY_LEARNING_RATE]

  @property
  def init_value(self):
    return self.configs[Constants.GROUP_PARAMS][Constants.KEY_PARAM_INIT]

  @property
  def clip_gradients(self):
    return self.configs[Constants.GROUP_PARAMS][Constants.KEY_CLIP_GRADIENTS]

  @property
  def regularization(self):
    return self.configs[Constants.GROUP_PARAMS][Constants.KEY_REGULARIZATION]

  @property
  def avg_loss_in_time(self):
    return self.configs[Constants.GROUP_PARAMS][
      Constants.KEY_AVERAGE_LOSS_IN_TIME]

  @property
  def decay_type(self):
    return self.configs[Constants.GROUP_PARAMS][Constants.KEY_DECAY_TYPE]

  @property
  def decay_rate(self):
    return self.configs[Constants.GROUP_PARAMS][Constants.KEY_DECAY_RATE]

  @property
  def decay_steps(self):
    return self.configs[Constants.GROUP_PARAMS][Constants.KEY_DECAY_STEPS]

  @property
  def start_decay_steps(self):
    return self.configs[Constants.GROUP_PARAMS][Constants.KEY_START_DECAY_STEPS]

  @property
  def min_learning_rate(self):
    return self.configs[Constants.GROUP_PARAMS][Constants.KEY_MIN_LEARNING_RATE]

  @property
  def beam_width(self):
    return self.configs[Constants.GROUP_PARAMS][Constants.KEY_BEAM_WIDTH]

  @property
  def replace_unknown_target(self):
    return self.configs[Constants.GROUP_PARAMS][
      Constants.KEY_REPLACE_UNKNOWN_TARGET]

  @property
  def training_features_file(self):
    return self.configs[Constants.GROUP_DATA][
      Constants.KEY_TRAINING_FEATURES_FILE]

  @property
  def training_labels_file(self):
    return self.configs[Constants.GROUP_DATA][
      Constants.KEY_TRAINING_LABELS_FILE]

  @property
  def eval_features_file(self):
    return self.configs[Constants.GROUP_DATA][
      Constants.KEY_EVALUATION_FEATURES_FILE]

  @property
  def eval_labels_file(self):
    return self.configs[Constants.GROUP_DATA][
      Constants.KEY_EVALUATION_LABELS_FILE]

  @property
  def features_vocab_file(self):
    return self.configs[Constants.GROUP_DATA][Constants.KEY_FEATURES_VOCAB_FILE]

  @property
  def labels_vocab_file(self):
    return self.configs[Constants.GROUP_DATA][Constants.KEY_LABELS_VOCAB_FILE]

  @property
  def max_features_len(self):
    return self.configs[Constants.GROUP_DATA][Constants.KEY_MAX_FEATURES_LEN]

  @property
  def max_labels_len(self):
    return self.configs[Constants.GROUP_DATA][Constants.KEY_MAX_LABELS_LEN]

  @property
  def out_dir(self):
    return self.configs[Constants.GROUP_DATA][Constants.KEY_OUT_DIR]

  @property
  def training_batch_size(self):
    return self.configs[Constants.GROUP_TRAIN][
      Constants.KEY_TRAINING_BATCH_SIZE]

  @property
  def save_checkpoints_steps(self):
    return self.configs[Constants.GROUP_TRAIN][
      Constants.KEY_SAVE_CHECKPOINTS_STEPS]

  @property
  def keep_max_checkpoints(self):
    return self.configs[Constants.GROUP_TRAIN][
      Constants.KEY_KEEP_MAX_CHECKPOINTS]

  @property
  def save_summary_steps(self):
    return self.configs[Constants.GROUP_TRAIN][Constants.KEY_SAVE_SUMMARY_STEPS]

  @property
  def training_steps(self):
    return self.configs[Constants.GROUP_TRAIN][Constants.KEY_TRAINING_STEPS]

  @property
  def bucket_width(self):
    return self.configs[Constants.GROUP_TRAIN][Constants.KEY_BUCKET_WIDTH]

  @property
  def shuffle_buffer_size(self):
    return self.configs[Constants.GROUP_TRAIN][
      Constants.KEY_SHUFFLE_BUFFER_SIZE]

  @property
  def training_prefetch_buffer_size(self):
    return self.configs[Constants.GROUP_TRAIN][
      Constants.KEY_TRAINING_PREFETCH_BUFFER_SIZE]

  @property
  def avg_last_checkpoints(self):
    return self.configs[Constants.GROUP_TRAIN][
      Constants.KEY_AVERAGE_LAST_CHECKPOINTS]

  @property
  def eval_batch_size(self):
    return self.configs[Constants.GROUP_EVAL][Constants.KEY_EVAL_BATCH_SIZE]

  @property
  def eval_prefetch_buffer_size(self):
    return self.configs[Constants.GROUP_EVAL][
      Constants.KEY_EVAL_PREFETCH_BUFFER_SIZE]

  @property
  def eval_delay(self):
    return self.configs[Constants.GROUP_EVAL][Constants.KEY_EVAL_DELAY]

  @property
  def save_eval_predictions(self):
    return self.configs[Constants.GROUP_EVAL][
      Constants.KEY_SAVE_EVAL_PREDICTIONS]

  @property
  def external_evaluators(self):
    return self.configs[Constants.GROUP_EVAL][Constants.KEY_EXTERNAL_EVALUATORS]

  @property
  def exporters(self):
    return self.configs[Constants.GROUP_EVAL][Constants.KEY_EXPORTERS]

  @property
  def infer_batch_size(self):
    return self.configs[Constants.GROUP_INFER][Constants.KEY_INFER_BATCH_SIZE]

  @property
  def infer_prefetch_size(self):
    return self.configs[Constants.GROUP_INFER][
      Constants.KEY_INFER_PREFETCH_BUFFER_SIZE]

  @property
  def n_best(self):
    return self.configs[Constants.GROUP_INFER][Constants.KEY_N_BEST]


config = Config()
