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

import abc
import argparse
import codecs
import os
import sys

import tensorflow as tf
from tensorflow.python.estimator.util import fn_args

from naivenmt.configs.arguments import add_arguments
from naivenmt.configs.hparams import Hparams
from naivenmt.hooks import CkptLoggingListener
from naivenmt.hooks import LifecycleLoggingHook
from naivenmt.hooks import SaveEvaluationPredictionsHook
from naivenmt.hooks import TrainTensorsSummaryHook
from naivenmt.models import BasicModel, AttentionModel, GNMTModel
from naivenmt.utils import average_ckpts


class NaiveNMTInterface(abc.ABC):

  @abc.abstractmethod
  def train(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def eval(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def predict(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def export(self):
    raise NotImplementedError()


class AbstractNaiveNMT(NaiveNMTInterface):

  def __init__(self, hparams):
    self.hparams = hparams
    self.model = self._create_model()
    self.estimator = self._create_estimator()

  @abc.abstractmethod
  def _create_train_hooks(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def _create_eval_hooks(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def _create_predict_hooks(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def _create_run_config(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def _create_session_config(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def _create_model_lifecycle_hooks(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def _create_model_tensors_hooks(self):
    raise NotImplementedError()

  def _create_estimator(self):
    return tf.estimator.Estimator(
      model_fn=self.model.model_fn(),
      model_dir=self.hparams.out_dir,
      config=self._create_run_config(),
      params=self.hparams)

  def _create_model(self):
    lifecycle_hooks = self._create_model_lifecycle_hooks()
    tensors_hooks = self._create_model_tensors_hooks()
    if not self.hparams.attention:
      return BasicModel(params=self.hparams,
                        predict_file=self.hparams.inference_input_file,
                        lifecycle_hooks=lifecycle_hooks,
                        tensors_hooks=tensors_hooks)
    if self.hparams.attention_architecture == "standard":
      return AttentionModel(params=self.hparams,
                            predict_file=self.hparams.inference_input_file,
                            lifecycle_hooks=lifecycle_hooks,
                            tensors_hooks=tensors_hooks)
    if self.hparams.attention_architecture in ["gnmt", "gnmt_v2"]:
      return GNMTModel(params=self.hparams,
                       predict_file=self.hparams.inference_input_file,
                       lifecycle_hooks=lifecycle_hooks,
                       tensors_hooks=tensors_hooks)
    raise ValueError("Can not create model.")

  def train(self):
    self.estimator.train(
      input_fn=self.model.input_fn(tf.estimator.ModeKeys.TRAIN),
      hooks=self._create_train_hooks(),
      max_steps=self.hparams.num_train_steps)
    if self.hparams.avg_ckpts:
      average_ckpts(self.hparams.out_dir, self.hparams.num_keep_ckpts,
                    self._create_session_config())

  def eval(self):
    self.estimator.evaluate(
      input_fn=self.model.input_fn(tf.estimator.ModeKeys.EVAL),
      hooks=self._create_eval_hooks(),
      checkpoint_path=None)

  def predict(self):
    infer_input_file = self.hparams.inference_input_file
    if not infer_input_file:
      raise ValueError("Inference input file must be provided.")
    infer_output_file = self.hparams.inference_output_file
    if not infer_output_file:
      infer_output_file = os.path.join(self.hparams.out_dir, "infer_output.txt")
    checkpoint_path = self.hparams.ckpt
    if not checkpoint_path:
      checkpoint_path = tf.train.latest_checkpoint(self.hparams.out_dir)
    predictions = self.estimator.predict(
      input_fn=self.model.input_fn(tf.estimator.ModeKeys.PREDICT),
      checkpoint_path=checkpoint_path,
      hooks=self._create_predict_hooks())

    with codecs.getwriter("utf-8")(
            tf.gfile.GFile(infer_output_file, mode="wb")) as fout:
      for prediction in predictions:
        fout.write((prediction + b'\n').decode("utf-8"))

  def export(self):
    # Use flags.ckpt to create ckpt for exporting
    checkpoint_path = self.hparams.ckpt
    if not checkpoint_path:
      checkpoint_path = tf.train.latest_checkpoint(self.hparams.out_dir)

    # TODO(luozhouyang) add an option to set export dir
    export_dir = os.path.join(self.estimator.model_dir, "export")
    if not os.path.isdir(export_dir):
      os.makedirs(export_dir)

    kwargs = {}
    if "strip_default_attrs" in fn_args(self.estimator.export_savedmodel):
      # Set strip_default_attrs to True for TensorFlow 1.6+ to stay consistent
      # with the behavior of tf.estimator.Exporter.
      kwargs["strip_default_attrs"] = True

    self.estimator.export_savedmodel(
      export_dir,
      serving_input_receiver_fn=self.model.serving_input_fn(),
      checkpoint_path=checkpoint_path,
      **kwargs)


class NaiveNMT(AbstractNaiveNMT):

  def _create_run_config(self):
    config = tf.estimator.RunConfig(
      model_dir=self.hparams.out_dir,
      session_config=self._create_session_config(),
      tf_random_seed=self.hparams.random_seed,
      save_summary_steps=100,
      save_checkpoints_steps=5000,
      save_checkpoints_secs=None,
      keep_checkpoint_max=10,
      log_step_count_steps=500)
    return config

  def _create_session_config(self):
    return tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False,
      gpu_options=tf.GPUOptions(allow_growth=False))

  def _create_model_lifecycle_hooks(self):
    return [LifecycleLoggingHook()]

  def _create_model_tensors_hooks(self):
    return [TrainTensorsSummaryHook()]

  def _create_train_hooks(self):
    step_count_hook = tf.train.StepCounterHook(
      every_n_steps=100, output_dir=self.hparams.out_dir)
    ckpt_saver_hook = tf.train.CheckpointSaverHook(
      save_steps=self.hparams,
      checkpoint_dir=self.hparams.out_dir,
      listeners=[CkptLoggingListener()])
    loss_nan_hook = tf.train.NanTensorHook(
      loss_tensor=tf.get_collection("loss"))
    train_hooks = [step_count_hook, ckpt_saver_hook, loss_nan_hook, ]
    return train_hooks

  def _create_eval_hooks(self):
    save_eval_predictions_hook = SaveEvaluationPredictionsHook(
      out_dir=self.hparams.out_dir)
    return [save_eval_predictions_hook]

  def _create_predict_hooks(self):
    return None


def main(unused_args):
  hparams = Hparams(FLAGS).build()
  naive_nmt = NaiveNMT(hparams)
  if FLAGS.mode == "train":
    naive_nmt.train()
  elif FLAGS.mode == "eval":
    naive_nmt.eval()
  elif FLAGS.mode == "predict":
    naive_nmt.predict()
  elif FLAGS.mode == "export":
    naive_nmt.export()
  else:
    raise ValueError("Unknown mode: %s" % FLAGS.mode)


FLAGS = None
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--mode", type=str,
                      choices=["train", "eval", "predict", "export"],
                      default="train",
                      help="Run mode.")
  add_arguments(parser)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
