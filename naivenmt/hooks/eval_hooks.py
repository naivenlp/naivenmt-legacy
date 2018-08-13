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

import os

import tensorflow as tf

from naivenmt.utils import get_dict_from_collection
from naivenmt.utils import get_predictions


class SaveEvaluationPredictionsHook(tf.train.SessionRunHook):
  """Do evaluation and save prediction results to file."""

  def __init__(self,
               out_dir,
               eos="</s>",
               subword_option="",
               post_evaluation_fn=None):
    """Init.

    Args:
      out_dir: model's dir
      eos: eos of params
      subword_option: subword options of params
      post_evaluation_fn: a callback fn with signature (global_steps, predictions_file),
        called after saving predictions
    """
    self.eos = eos
    self.subword_option = subword_option
    self.output_file = os.path.join(out_dir, "output_dev")
    self.post_evaluation_fn = post_evaluation_fn
    self.predictions = None
    self.global_steps = None

  def begin(self):
    self.predictions = get_dict_from_collection("predictions")
    self.global_steps = tf.train.get_global_step()

  def before_run(self, run_context):
    if not self.predictions:
      raise ValueError("Model does not define predictions.")
    if not self.global_steps:
      raise ValueError("Not created global steps.")
    return tf.train.SessionRunArgs([self.predictions, self.global_steps])

  def after_run(self,
                run_context,  # pylint: disable=unused-argument
                run_values):
    predictions, self.global_steps = run_values.results
    self.output_file = self.output_file + "." + self.global_steps
    predictions = get_predictions(predictions, self.eos, self.subword_option)

    with open(self.output_file, mode="a", encoding="utf8") as f:
      if isinstance(predictions, str):
        f.write(predictions + "\n")
      elif isinstance(predictions, list):
        for p in predictions:
          f.write(p + "\n")

  def end(self, session):
    tf.logging.info("Evaluation predictions saved to %s" % self.output_file)
    if self.post_evaluation_fn:
      self.post_evaluation_fn(self.global_steps, self.output_file)
