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

from naivenmt.configs.hparams import Hparams
from naivenmt.configs.arguments import add_arguments
from naivenmt.hooks import TensorSummaryHook
from naivenmt.hooks import TensorsCollectionHook
from naivenmt.models import BasicModel, AttentionModel, GNMTModel
from naivenmt.hooks import CkptLoggingListener
from naivenmt.hooks import LifecycleLoggingHook


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


class NaiveNMT(NaiveNMTInterface):

  def __init__(self, hparams):
    self.hparams = hparams
    self.model = self._create_model()
    self.estimator = self._create_estimator()

  def _create_estimator(self):
    sess_config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False,
      gpu_options=tf.GPUOptions(
        allow_growth=False))

    # TODO(luozhouyang) set configs to sess in models
    run_configs = tf.estimator.RunConfig(
      model_dir=self.hparams.out_dir,
      session_config=sess_config,
      tf_random_seed=self.hparams.random_seed)

    return tf.estimator.Estimator(
      model_fn=self.model.model_fn(),
      model_dir=self.hparams.out_dir,
      config=run_configs,
      params=self.hparams)

  def _create_model(self):
    lifecycle_hooks = [LifecycleLoggingHook()]
    tensors_hooks = [TensorSummaryHook(), TensorsCollectionHook(), ]
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
    # TODO(luozhouyang) add hooks
    train_hooks = [tf.train.StepCounterHook(every_n_steps=100,
                                            output_dir=self.hparams.out_dir),
                   tf.train.CheckpointSaverHook(
                     save_steps=100,
                     checkpoint_dir=self.hparams.out_dir,
                     listeners=[CkptLoggingListener()]),
                   tf.train.NanTensorHook(tf.get_collection("loss")),
                   # tf.train.SummarySaverHook(save_steps=100,
                   #                           output_dir=self.hparams.out_dir)
                   ]
    train_spec = tf.estimator.TrainSpec(
      input_fn=self.model.input_fn(tf.estimator.ModeKeys.TRAIN),
      max_steps=self.hparams.num_train_steps,
      hooks=train_hooks)

    self.estimator.train(
      input_fn=train_spec.input_fn,
      hooks=train_spec.hooks,
      max_steps=train_spec.max_steps)
    # TODO(luozhouyang) average ckpts

  def eval(self):
    # TODO(luozhouyang) add hooks and set checkpoint_path
    eval_hooks = []
    self.estimator.evaluate(
      input_fn=self.model.input_fn(tf.estimator.ModeKeys.EVAL),
      hooks=eval_hooks,
      checkpoint_path=None)

  def predict(self):
    infer_input_file = self.hparams.inference_input_file
    if not infer_input_file:
      raise ValueError("Inference input file must be provided.")
    infer_output_file = self.hparams.inference_output_file
    if not infer_output_file:
      infer_output_file = os.path.join(self.hparams.out_dir, "infer_output.txt")
    # TODO(luozhouyang) add option to set ckpt
    checkpoint_path = tf.train.latest_checkpoint(self.hparams.out_dir)
    # TODO(luozhouyang) add infer hooks
    infer_hooks = []
    predictions = self.estimator.predict(
      input_fn=self.model.input_fn(tf.estimator.ModeKeys.PREDICT),
      checkpoint_path=checkpoint_path,
      hooks=infer_hooks)

    with codecs.getwriter("utf-8")(
            tf.gfile.GFile(infer_output_file, mode="wb")) as fout:
      for prediction in predictions:
        fout.write((prediction + b'\n').decode("utf-8"))

  def export(self):
    # TODO(luozhouyang) add option to set ckpt
    checkpoint_path = tf.train.latest_checkpoint(self.hparams.out_dir)

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
