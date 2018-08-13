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


class ModelTensorsHook(abc.ABC):
  """Listen tensors state when it has been created."""

  @abc.abstractmethod
  def on_global_steps_created(self, global_steps):
    """Listen global_steps.

    Args:
      global_steps: global steps.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def on_learning_rate_created(self, learning_rate):
    """Listen learning_rate.

    Args:
      learning_rate: learning rate.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def on_gradients_created(self, gradients):
    """Listen gradients before clipping.

    Args:
      gradients: gradients.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def on_gradients_clipped(self, clipped_grads, grad_norm):
    """Listen states of clipped gradients.

    Args:
      clipped_grads: clipped gradients.
      grad_norm: gradients norm.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def on_optimizer_created(self, optimizer):
    """Listen optimizer.

    Args:
      optimizer: optimizer
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def on_loss_created(self, loss):
    """Listen loss.

    Args:
      loss
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def end(self):
    """Called after all above tensors being created."""
    raise NotImplementedError()


class ModelLifecycleHook(abc.ABC):
  """Listeners to listen model's lifecycle.

  Generally speaking, we can just get information from tensors,
  but not to update these tensors.
  """

  @abc.abstractmethod
  def before_encode(self, mode, features):
    """Listen tensors states before encode.

    Args:
      mode: mode, one of estimator's ModeKeys.
      features: features inputs, instance of ``naivenmt.inputters.Features``.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def after_encode(self, mode, features, outputs, state):
    """Listen tensors states after encode, before decode.

    Args:
      mode: mode, one of estimator's ModeKeys.
      features: features inputs, instance of ``naivenmt.inputters.Features``.
      outputs: encoder's outputs.
      state: encoder's output state.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def before_decode(self, mode, outputs, state, labels, src_seq_len):
    """Listen tensors state before decode.

    Args:
      mode: mode, one of estimator's ModeKeys.
      outputs: encoder's outputs
      state: encoder's output state
      labels: labels inputs, instance of ``naivenmt.inputters.Labels``.
      src_seq_len: source sequence length
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def after_decode(self, mode, logits, loss, sample_id, final_context_state):
    """Listen tensors state after decode.

    Args:
      mode: mode, one of estimator's ModeKeys.
      logits: logits
      loss: loss, None for PREDICT mode.
      sample_id: sample id results of decode.
      final_context_state: state of decoder.
    """
    raise NotImplementedError()


class LifecycleLoggingHook(ModelLifecycleHook):

  def before_encode(self, mode, features):
    tf.logging.info("Model in mode %s is about to encoding inputs: %s."
                    % (mode, features.source_ids))

  def after_encode(self, mode, features, outputs, state):
    tf.logging.info("Decoder outputs: %s" % outputs)
    tf.logging.info("Decoder state  : %s" % state)

  def before_decode(self, mode, outputs, state, labels, src_seq_len):
    tf.logging.info("Model in mode %s is about to decoding.")
    tf.logging.info("Target inputs : %s" % labels.target_input_ids)
    tf.logging.info("Target outputs: %s" % labels.target_output_ids)

  def after_decode(self, mode, logits, loss, sample_id, final_context_state):
    tf.logging.info("Model decode loss     : %s" % logits)
    tf.logging.info("Model decode sample id: %s" % sample_id)
    tf.logging.info("Model decode state    : %s" % final_context_state)


class TrainTensorsSummaryHook(ModelTensorsHook):
  """Summary tensors."""

  def __init__(self):
    self.summaries = {}

  def on_global_steps_created(self, global_steps):
    self.summaries["global_steps"] = global_steps

  def on_learning_rate_created(self, learning_rate):
    self.summaries["learning_rate"] = learning_rate

  def on_gradients_created(self, gradients):
    self.summaries["gradients"] = gradients

  def on_gradients_clipped(self, clipped_grads, grad_norm):
    self.summaries["clipped_grads"] = tf.global_norm(clipped_grads)
    self.summaries["grad_norm"] = grad_norm

  def on_optimizer_created(self, optimizer):
    pass

  def on_loss_created(self, loss):
    if not loss:  # predict mode
      return
    # TODO(luozhouyang) need a `mode` arg?
    self.summaries["loss"] = loss

  def end(self):
    summaries = []
    for k, v in self.summaries.items():
      summary = tf.summary.scalar(k, v)
      summaries.append(summary)
    tf.summary.merge(summaries, name="train_summary")
    tf.logging.info("Tensors summaries are merged.")
