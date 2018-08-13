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
import numpy as np


def average_ckpts(out_dir, num_ckpts=5, sess_config=None):
  """Average checkpoints for better performance.

  Args:
    out_dir: models' dir
    num_ckpts: number of checkpoints to average
    sess_config: configs for session
  """
  checkpoints_path = tf.train.get_checkpoint_state(
    out_dir).all_model_checkpoint_paths

  if len(checkpoints_path) > num_ckpts:
    checkpoints_path = checkpoints_path[-num_ckpts:]
  num_ave_ckpts = len(checkpoints_path)
  tf.logging.info("Averaging %d checkpoints." % num_ave_ckpts)

  var_list = tf.train.list_variables(checkpoints_path[0])
  avg_values = {}
  for name, shape in var_list:
    if not name.startswith("global_step"):
      avg_values[name] = np.zeros(shape, dtype=np.float32)

  for checkpoint_path in checkpoints_path:
    tf.logging.info("Loading checkpoint %s" % checkpoint_path)
    reader = tf.train.load_checkpoint(checkpoint_path)
    for name in avg_values:
      avg_values[name] += reader.get_tensor(name) / num_ave_ckpts
