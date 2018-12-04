# Copyright 2018 luozhouyang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import functools
import json
import os

import tensorflow as tf

from naivenmt.configs import HParamsBuilder
from naivenmt.hooks import CountParamsHook
from naivenmt.hooks import InitHook
from naivenmt.hooks import SaveEvaluationPredictionsHook
from naivenmt.models import AttentionModel
from naivenmt.models import BasicModel
from naivenmt.models import GNMTModel
from naivenmt.utils import text_utils


class NaiveNMT(object):

  def __init__(self, hparams, model):
    self.hparams = hparams
    self.model = model

    sess_config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False,
      gpu_options=tf.GPUOptions(allow_growth=True))

    # TODO(luozhouyang) Add distribution strategy
    run_config = tf.estimator.RunConfig(
      model_dir=self.hparams.out_dir,
      session_config=sess_config,
      tf_random_seed=self.hparams.random_seed,
      save_checkpoints_secs=None,
      save_checkpoints_steps=self.hparams.save_ckpt_steps,
      keep_checkpoint_max=self.hparams.keep_ckpt_max,
      log_step_count_steps=self.hparams.log_step_count_steps,
      train_distribute=None)

    self.estimator = tf.estimator.Estimator(
      model_fn=self.model.model_fn,
      model_dir=self.hparams.out_dir,
      params=self.hparams,
      config=run_config)

  def train(self):
    train_hooks = self._build_train_hooks()
    self.estimator.train(
      input_fn=functools.partial(self.model.input_fn,
                                 self.hparams,
                                 tf.estimator.ModeKeys.TRAIN),
      hooks=train_hooks,
      max_steps=self.hparams.train_steps)

  def eval(self):
    eval_hooks = self._build_eval_hooks()
    self.estimator.evaluate(
      input_fn=functools.partial(self.model.input_fn,
                                 self.hparams,
                                 tf.estimator.ModeKeys.EVAL),
      hooks=eval_hooks)

  def train_and_eval(self):
    train_hooks = self._build_train_hooks()
    eval_hooks = self._build_eval_hooks()
    train_spec = tf.estimator.TrainSpec(
      input_fn=functools.partial(self.model.input_fn,
                                 self.hparams,
                                 tf.estimator.ModeKeys.TRAIN),
      hooks=train_hooks,
      max_steps=self.hparams.train_steps)
    # TODO(luozhouyang) Add exporters
    eval_spec = tf.estimator.EvalSpec(
      input_fn=functools.partial(self.model.input_fn,
                                 self.hparams,
                                 tf.estimator.ModeKeys.EVAL),
      hooks=eval_hooks)
    tf.estimator.train_and_evaluate(
      self.estimator,
      train_spec=train_spec,
      eval_spec=eval_spec)

  def predict(self):
    predict_hooks = self._build_predict_hooks()
    predictions = self.estimator.predict(
      input_fn=functools.partial(self.model.input_fn,
                                 self.hparams,
                                 tf.estimator.ModeKeys.PREDICT),
      hooks=predict_hooks)
    # TODO(luozhouyang) save prediction results
    results = text_utils.get_predictions(predictions,
                                         self.hparams.eos,
                                         self.hparams.subword_option)
    print(results)

  def export(self):
    # TODO(luozhouyang) Add export ckpt path in hparams
    self.estimator.export_savedmodel(
      export_dir_base=os.path.join(self.hparams.out_dir, "export"),
      serving_input_receiver_fn=self.model.serving_input_receiver_fn,
      checkpoint_path=None)

  def _build_train_hooks(self):
    hooks = [InitHook(), CountParamsHook()]
    return hooks

  def _build_eval_hooks(self):
    hooks = [InitHook()]
    save_path = os.path.join(self.estimator.model_dir, "eval")
    if not os.path.exists(save_path):
      os.makedirs(save_path)
    output_file = os.path.join(save_path, "predictions.txt")
    hooks.append(SaveEvaluationPredictionsHook(
      output_file=output_file,
      eos=self.hparams.eos,
      subword_option=self.hparams.subword_option))
    return hooks

  def _build_predict_hooks(self):
    hooks = [InitHook()]
    return hooks


def create_model(m, params):
  if m == "basic_model":
    return BasicModel(params=params)
  elif m == "attention_model":
    return AttentionModel(params=params)
  elif m == "gnmt_model":
    return GNMTModel(params=params)
  else:
    raise ValueError("Invalid model type %s" % m)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)

  parser = argparse.ArgumentParser()
  parser.add_argument("--mode", type=str,
                      choices=["train", "eval", "train_and_eval", "predict",
                               "export"],
                      default="train",
                      help="Run mode.")
  parser.add_argument("--model", type=str,
                      choices=["basic_model", "attention_model", "gnmt_model"],
                      default="basic_model",
                      help="The model you want to use.")
  parser.add_argument("--params_file", type=str,
                      required=True,
                      help="Params config file in JSON format.")
  args, _ = parser.parse_known_args()
  mode = args.mode
  with open(args.params_file, mode="rt", encoding="utf8") as f:
    configs = json.loads(f)
  hparams = HParamsBuilder(dict_config=configs).build()
  model = create_model(args.model, hparams)
  naivenmt = NaiveNMT(hparams=hparams, model=model)
  if mode == "train":
    naivenmt.train()
  elif mode == "eval":
    naivenmt.eval()
  elif mode == "train_and_eval":
    naivenmt.train_and_eval()
  elif mode == "predict":
    naivenmt.predict()
  elif model == "export":
    naivenmt.export()
  else:
    raise ValueError("Invalid mode %s" % mode)
