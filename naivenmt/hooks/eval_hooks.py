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

import tensorflow as tf

from naivenmt.utils import collection_utils
from naivenmt.utils import text_utils


class SaveEvaluationPredictionsHook(tf.train.SessionRunHook):
  """Do evaluation and save prediction results to file."""

  def __init__(self,
               output_file,
               eos="</s>",
               subword_option=""):
    self.eos = eos
    self.subword_option = subword_option
    self.output_file = output_file
    self.predictions = None
    self.global_steps = None
    self.output_path = None

  def begin(self):
    self.predictions = collection_utils.get_dict_from_collection(
      name=collection_utils.PREDICTIONS)
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
    predictions, global_steps = run_values.results
    predictions = text_utils.get_predictions(
      predictions, self.eos, self.subword_option)

    self.output_path = "{}.{}".format(self.output_file, global_steps)

    with open(self.output_path, mode="a", encoding="utf8") as f:
      if isinstance(predictions, str):
        f.write(predictions + "\n")
      elif isinstance(predictions, list):
        for p in predictions:
          f.write(p + "\n")

  def end(self, session):
    tf.logging.info("Evaluation predictions saved to %s" % self.output_path)
