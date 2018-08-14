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

import numpy as np
import six
import tensorflow as tf


# This script is modified version of OpenNMT-tf
# https://github.com/OpenNMT/OpenNMT-tf/blob/master/opennmt/utils/checkpoint.py
# which is modified version of tensorflow/tensor2tensor
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/bin/t2t_avg_all.py
# tensorflow/tensor2tensor comes with the following license and copyright notice:
#
# Copyright 2017 The Tensor2Tensor Authors.
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
def average_ckpts(out_dir, num_ckpts=5, sess_config=None):
  """Average checkpoints for better performance.

  Args:
    out_dir: models' dir
    num_ckpts: number of checkpoints to average
    sess_config: configs for session
  """
  avg_out_dir = os.path.join(out_dir, "avg_checkpoints")
  if not tf.gfile.Exists(avg_out_dir):
    tf.gfile.MakeDirs(avg_out_dir)

  checkpoints_path = tf.train.get_checkpoint_state(
    out_dir).all_model_checkpoint_paths

  if len(checkpoints_path) > num_ckpts:
    checkpoints_path = checkpoints_path[-num_ckpts:]
  if len(checkpoints_path) == 0:
    tf.logging.warn("No checkpoints path found. Averaging cancel.")
    return
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

  return _save_new_variables(avg_values,
                             avg_out_dir,
                             checkpoints_path[-1],
                             session_config=sess_config)


# This script is most copied from
# https://github.com/OpenNMT/OpenNMT-tf/blob/master/opennmt/utils/checkpoint.py
def _save_new_variables(variables,
                        output_dir,
                        base_checkpoint_path,
                        session_config=None):
  if "global_step" in variables:
    del variables["global_step"]
  tf_vars = []
  for name, value in six.iteritems(variables):
    trainable = True
    dtype = tf.as_dtype(value.dtype)
    if name.startswith("words_per_sec"):
      trainable = False
      dtype = tf.int64  # TODO: why is the dtype not correct for these variables?
    tf_vars.append(tf.get_variable(
      name,
      shape=value.shape,
      dtype=dtype,
      trainable=trainable))
  placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
  assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]

  latest_step = int(base_checkpoint_path.split("-")[-1])
  out_base_file = os.path.join(output_dir, "translate.ckpt")
  global_step = tf.get_variable(
    "global_step",
    initializer=tf.constant(latest_step, dtype=tf.int64),
    trainable=False)
  saver = tf.train.Saver(tf.global_variables())

  with tf.Session(config=session_config) as sess:
    sess.run(tf.global_variables_initializer())
    for p, assign_op, (name, value) in zip(placeholders, assign_ops,
                                           six.iteritems(variables)):
      sess.run(assign_op, {p: value})
    tf.logging.info("Saving new checkpoint to %s" % output_dir)
    saver.save(sess, out_base_file, global_step=global_step)

  return output_dir
