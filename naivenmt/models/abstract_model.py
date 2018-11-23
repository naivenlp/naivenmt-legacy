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
import tensorflow as tf


class AbstractModel(abc.ABC):
  """Abstract model."""

  def input_fn(self, params, mode):
    """Build input_fn for estimator.

    Args:
      params: A python object, hyper params
      mode. A scalar, one of `tf.estimator.ModeKeys`

    Returns:
      A (features, labels) tuple.
    """
    raise NotImplementedError()

  def model_fn(self, features, labels, mode, params, config=None):
    """Build model_fn for estimator."""
    raise NotImplementedError()

  def serving_input_receiver_fn(self):
    """Build input receiver fn for exporting saved model."""
    receiver_tensors = {
      "inputs": tf.placeholder(dtype=tf.string, shape=(None, None)),
      "inputs_length": tf.placeholder(dtype=tf.int32, shape=(None))
    }
    features = receiver_tensors.copy()
    return tf.estimator.export.ServingInputReceiver(
      features=features,
      receiver_tensors=receiver_tensors)
