import os

import yaml

from .contants import Constants
from .data_config import DataConfigInterface
from .evaluation_config import EvaluationConfigInterface
from .inference_config import InferenceConfigInterface
from .params_config import ParamsConfigInterface
from .training_config import TrainingConfigInterface


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

  def get_optimizer(self):
    return self.configs[Constants.GROUP_PARAMS][Constants.KEY_OPTIMIZER]

  def get_learning_rate(self):
    return self.configs[Constants.GROUP_PARAMS][Constants.KEY_LEARNING_RATE]

  def get_init_value(self):
    return self.configs[Constants.GROUP_PARAMS][Constants.KEY_PARAM_INIT]

  def get_clip_gradients(self):
    return self.configs[Constants.GROUP_PARAMS][Constants.KEY_CLIP_GRADIENTS]

  def get_regularization(self):
    return self.configs[Constants.GROUP_PARAMS][Constants.KEY_REGULARIZATION]

  def get_avg_loss_in_time(self):
    return self.configs[Constants.GROUP_PARAMS][
      Constants.KEY_AVERAGE_LOSS_IN_TIME]

  def get_decay_type(self):
    return self.configs[Constants.GROUP_PARAMS][Constants.KEY_DECAY_TYPE]

  def get_decay_rate(self):
    return self.configs[Constants.GROUP_PARAMS][Constants.KEY_DECAY_RATE]

  def get_decay_steps(self):
    return self.configs[Constants.GROUP_PARAMS][Constants.KEY_DECAY_STEPS]

  def get_start_decay_steps(self):
    return self.configs[Constants.GROUP_PARAMS][Constants.KEY_START_DECAY_STEPS]

  def get_min_learning_rate(self):
    return self.configs[Constants.GROUP_PARAMS][Constants.KEY_MIN_LEARNING_RATE]

  def get_beam_width(self):
    return self.configs[Constants.GROUP_PARAMS][Constants.KEY_BEAM_WIDTH]

  def get_replace_unknown_target(self):
    return self.configs[Constants.GROUP_PARAMS][
      Constants.KEY_REPLACE_UNKNOWN_TARGET]

  def get_training_features_file(self):
    return self.configs[Constants.GROUP_DATA][
      Constants.KEY_TRAINING_FEATURES_FILE]

  def get_training_labels_file(self):
    return self.configs[Constants.GROUP_DATA][
      Constants.KEY_TRAINING_LABELS_FILE]

  def get_eval_features_file(self):
    return self.configs[Constants.GROUP_DATA][
      Constants.KEY_EVALUATION_FEATURES_FILE]

  def get_eval_labels_file(self):
    return self.configs[Constants.GROUP_DATA][
      Constants.KEY_EVALUATION_LABELS_FILE]

  def get_features_vocab_file(self):
    return self.configs[Constants.GROUP_DATA][Constants.KEY_FEATURES_VOCAB_FILE]

  def get_labels_vocab_file(self):
    return self.configs[Constants.GROUP_DATA][Constants.KEY_LABELS_VOCAB_FILE]

  def get_max_features_len(self):
    return self.configs[Constants.GROUP_DATA][Constants.KEY_MAX_FEATURES_LEN]

  def get_max_labels_len(self):
    return self.configs[Constants.GROUP_DATA][Constants.KEY_MAX_LABELS_LEN]

  def get_out_dir(self):
    return self.configs[Constants.GROUP_DATA][Constants.KEY_OUT_DIR]

  def get_training_batch_size(self):
    return self.configs[Constants.GROUP_TRAIN][
      Constants.KEY_TRAINING_BATCH_SIZE]

  def get_save_checkpoints_steps(self):
    return self.configs[Constants.GROUP_TRAIN][
      Constants.KEY_SAVE_CHECKPOINTS_STEPS]

  def get_keep_max_checkpoints(self):
    return self.configs[Constants.GROUP_TRAIN][
      Constants.KEY_KEEP_MAX_CHECKPOINTS]

  def get_save_summary_steps(self):
    return self.configs[Constants.GROUP_TRAIN][Constants.KEY_SAVE_SUMMARY_STEPS]

  def get_training_steps(self):
    return self.configs[Constants.GROUP_TRAIN][Constants.KEY_TRAINING_STEPS]

  def get_bucket_width(self):
    return self.configs[Constants.GROUP_TRAIN][Constants.KEY_BUCKET_WIDTH]

  def get_shuffle_buffer_size(self):
    return self.configs[Constants.GROUP_TRAIN][
      Constants.KEY_SHUFFLE_BUFFER_SIZE]

  def get_training_prefetch_buffer_size(self):
    return self.configs[Constants.GROUP_TRAIN][
      Constants.KEY_TRAINING_PREFETCH_BUFFER_SIZE]

  def get_avg_last_checkpoints(self):
    return self.configs[Constants.GROUP_TRAIN][
      Constants.KEY_AVERAGE_LAST_CHECKPOINTS]

  def get_eval_batch_size(self):
    return self.configs[Constants.GROUP_EVAL][Constants.KEY_EVAL_BATCH_SIZE]

  def get_eval_prefetch_buffer_size(self):
    return self.configs[Constants.GROUP_EVAL][
      Constants.KEY_EVAL_PREFETCH_BUFFER_SIZE]

  def get_eval_delay(self):
    return self.configs[Constants.GROUP_EVAL][Constants.KEY_EVAL_DELAY]

  def get_save_eval_predictions(self):
    return self.configs[Constants.GROUP_EVAL][
      Constants.KEY_SAVE_EVAL_PREDICTIONS]

  def get_external_evaluators(self):
    return self.configs[Constants.GROUP_EVAL][Constants.KEY_EXTERNAL_EVALUATORS]

  def get_exporters(self):
    return self.configs[Constants.GROUP_EVAL][Constants.KEY_EXPORTERS]

  def get_infer_batch_size(self):
    return self.configs[Constants.GROUP_INFER][Constants.KEY_INFER_BATCH_SIZE]

  def get_infer_prefetch_size(self):
    return self.configs[Constants.GROUP_INFER][
      Constants.KEY_INFER_PREFETCH_BUFFER_SIZE]

  def get_n_best(self):
    return self.configs[Constants.GROUP_INFER][Constants.KEY_N_BEST]


config = Config()
