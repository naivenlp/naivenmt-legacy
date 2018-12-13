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

from naivenmt.tests import common_test_utils as utils

NUM_LAYERS_2 = 2
NUM_LAYERS_4 = 4


def parse_func(unit_type):
  if unit_type == "lstm":
    func = utils.get_uni_lstm_encoder_results
  elif unit_type == "layer_norm_lstm":
    func = utils.get_uni_layer_norm_lstm_encoder_results
  elif unit_type == "nas":
    func = utils.get_uni_nas_encoder_results
  else:
    raise ValueError("Invalid unit_type %s" % unit_type)
  return func


class BasicDecodersTrainOrEvalTest(tf.test.TestCase):
  """Test basic decoders in TRAIN ot EVAL mode. There has no difference in
    bi-encoder or uni-encoder, because the outputs and states keeps the same
    format in this two types of encoders. time_major=True."""

  def runGRUDecoder(
          self, decoder, num_layers, encoder_outputs,
          encoder_states, labels, src_seq_len):
    for mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
      logits, predict_ids, states = decoder.decode(
        mode=mode,
        encoder_outputs=encoder_outputs,
        encoder_state=encoder_states,
        labels=labels,
        src_seq_len=src_seq_len)

      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())
        logits, predict_ids, states = sess.run(
          [logits, predict_ids, states]
        )
        print(logits.shape)
        print(predict_ids.shape)
        print(predict_ids)
        self.assertEqual(tf.int32, predict_ids.dtype)
        # TODO(luozhouyang) why logits.shape[0] is not certain? 0/1/2
        self.assertEqual(logits.shape[0], predict_ids.shape[0])

        # final states is a tuple of (states_c, states_h) of length num_layers
        self.assertEqual(num_layers, len(states))
        for i in range(num_layers):
          self.assertAllEqual([utils.BATCH_SIZE, utils.DEPTH], states[i].shape)

  def runLSTMDecoder(
          self, decoder, num_layers, encoder_outputs,
          encoder_states, labels, src_seq_len):
    for mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
      logits, predict_ids, final_states = decoder.decode(
        mode=mode,
        encoder_outputs=encoder_outputs,
        encoder_state=encoder_states,
        labels=labels,
        src_seq_len=src_seq_len)

      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())
        logits, predict_ids, final_states = sess.run(
          [logits, predict_ids, final_states]
        )
        print(logits.shape)
        print(predict_ids.shape)
        print(predict_ids)

        self.assertEqual(tf.int32, predict_ids.dtype)
        # TODO(luozhouyang) why logits.shape[0] is not certain? 0/1/2
        self.assertEqual(logits.shape[0], predict_ids.shape[0])

        # final states is a tuple of (states_c, states_h) of length num_layers
        self.assertEqual(num_layers, len(final_states))
        for i in range(num_layers):
          # states_c
          self.assertAllEqual(
            [utils.BATCH_SIZE, utils.DEPTH], final_states[i][0].shape)
          # states_h
          self.assertAllEqual(
            [utils.BATCH_SIZE, utils.DEPTH], final_states[i][1].shape)

  def _testBasicLSTMLikeDecoder(self, unit_type):
    for num_layers in range(1, 10):
      configs = {
        "num_encoder_layers": num_layers,
        "num_decoder_layers": num_layers,
        "encoder_type": "uni",
        "unit_type": unit_type,
        "time_major": True
      }
      decoder = utils.build_basic_decoder(configs)
      # target_vocab_size if 6, which determines the shape of logits
      self.assertEqual(6, decoder.target_vocab_size)
      func = parse_func(unit_type)
      enc_outputs, enc_outputs_len, enc_states = func(num_layers)
      self.runLSTMDecoder(
        decoder, num_layers, enc_outputs,
        enc_states, utils.get_labels(), enc_outputs_len)

  def _testBiEncoderBasicLSTMLikeDecoder(self, unit_type):
    for num_layers in range(2, 10, 2):
      configs = {
        "num_encoder_layers": num_layers,
        "num_decoder_layers": num_layers,
        "encoder_type": "bi",
        "unit_type": unit_type,
        "time_major": True
      }
      decoder = utils.build_basic_decoder(configs)
      # target_vocab_size if 6, which determines the shape of logits
      self.assertEqual(6, decoder.target_vocab_size)
      func = parse_func(unit_type)
      enc_outputs, enc_outputs_len, enc_states = func(num_layers)
      self.runLSTMDecoder(
        decoder, num_layers, enc_outputs,
        enc_states, utils.get_labels(), enc_outputs_len)

  def testBasicLSTMDecoder(self):
    self._testBasicLSTMLikeDecoder("lstm")

  def testBiEncoderBasicLSTMDecoder(self):
    self._testBiEncoderBasicLSTMLikeDecoder("lstm")

  def testBasicLayerNormLSTMDecoder(self):
    self._testBasicLSTMLikeDecoder("layer_norm_lstm")

  def testBiEncoderBasicLayerNormLSTMDecoder(self):
    self._testBiEncoderBasicLSTMLikeDecoder("layer_norm_lstm")

  def testBasicNASDecoder(self):
    self._testBasicLSTMLikeDecoder("nas")

  def testBiEncoderBasicNASDecoder(self):
    self._testBiEncoderBasicLSTMLikeDecoder("nas")

  def testBasicGRUDecoder(self):
    for num_layers in range(1, 10):
      configs = {
        "num_encoder_layers": num_layers,
        "num_decoder_layers": num_layers,
        "encoder_type": "uni",
        "unit_type": "gru",
        "time_major": True
      }
      decoder = utils.build_basic_decoder(configs)
      enc_outputs, enc_outputs_len, enc_states = (
        utils.get_uni_gru_encoder_results(num_layers))
      self.runGRUDecoder(
        decoder, num_layers, enc_outputs,
        enc_states, utils.get_labels(), enc_outputs_len)

  def testBiEncoderBasicGRUDecoder(self):
    for num_layers in range(2, 10, 2):
      configs = {
        "num_encoder_layers": num_layers,
        "num_decoder_layers": num_layers,
        "encoder_type": "bi",
        "unit_type": "gru",
        "time_major": True
      }
      decoder = utils.build_basic_decoder(configs)
      enc_outputs, enc_outputs_len, enc_states = (
        utils.get_bi_gru_encoder_results(num_layers))
      self.runGRUDecoder(
        decoder, num_layers, enc_outputs,
        enc_states, utils.get_labels(), enc_outputs_len)


class BasicDecodersPredictTest(tf.test.TestCase):
  """Test basic decoders in PREDICT mode."""

  def runLSTMDecoder(
          self, decoder, num_layers, outputs,
          states, labels, src_seq_len):
    logits, predict_ids, states = decoder.decode(
      mode=tf.estimator.ModeKeys.PREDICT,
      encoder_outputs=outputs,
      encoder_state=states,
      labels=labels,
      src_seq_len=src_seq_len)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      logits, predict_ids, states = sess.run(
        [logits, predict_ids, states]
      )
      print(logits.shape)
      print(predict_ids.shape)
      print(predict_ids)
      self.assertEqual(tf.int32, predict_ids.dtype)
      # TODO(luozhouyang) why logits.shape[0] is not certain? 0/1/2
      self.assertEqual(logits.shape[0], predict_ids.shape[0])

      # final states is a tuple of (states_c, states_h) of length num_layers
      self.assertEqual(num_layers, len(states))
      for i in range(num_layers):
        self.assertAllEqual([utils.BATCH_SIZE, utils.DEPTH], states[i].shape)

  def runGRUDecoder(self, decoder, num_layers):
    pass

  def _testLSTMLikeGreedyDecoder(self, unit_type):
    for num_layers in range(1, 10):
      configs = {
        "unit_type": unit_type,
        "encoder_type": "uni",
        "num_encoder_layers": num_layers,
        "num_decoder_layers": num_layers,
        "forget_bias": 1.0,
        "beam_width": 0,
        'sampling_temperature': 0.0,
        "time_major": True,
      }
      decoder = utils.build_basic_decoder(configs)
      func = parse_func(unit_type)
      outputs, output_length, states = func(num_layers)
      self.runLSTMDecoder(
        decoder, num_layers, outputs,
        states, utils.get_labels(), output_length)

  def _testLSTMLikeSamplingDecoder(self, unit_type):
    for num_layers in range(1, 10):
      configs = {
        "unit_type": unit_type,
        "encoder_type": "uni",
        "num_encoder_layers": num_layers,
        "num_decoder_layers": num_layers,
        "forget_bias": 1.0,
        "beam_width": 0,
        'sampling_temperature': 0.5,
        "time_major": True,
      }
      decoder = utils.build_basic_decoder(configs)
      func = parse_func(unit_type)
      outputs, output_length, states = func(num_layers)
      self.runLSTMDecoder(
        decoder, num_layers, outputs,
        states, utils.get_labels(), output_length)

  def _testLSTMLikeBeamDecoder(self, unit_type):
    for num_layers in range(1, 10):
      configs = {
        "unit_type": unit_type,
        "encoder_type": "uni",
        "num_encoder_layers": num_layers,
        "num_decoder_layers": num_layers,
        "forget_bias": 1.0,
        "beam_width": 5,
        'sampling_temperature': 0.0,
        "time_major": True,
      }
      decoder = utils.build_basic_decoder(configs)
      func = parse_func(unit_type)
      outputs, output_length, states = func(num_layers)
      self.runLSTMDecoder(
        decoder, num_layers, outputs,
        states, utils.get_labels(), output_length)

  def testBasicLSTMGreedyDecoder(self):
    self._testLSTMLikeGreedyDecoder("lstm")

  def testBasicLSTMSamplingDecoder(self):
    self._testLSTMLikeSamplingDecoder("lstm")

  def testBasicLSTMBeamDecoder(self):
    self._testLSTMLikeBeamDecoder("lstm")

  def testBasicLayerNormLSTMGreedyDecoder(self):
    self._testLSTMLikeGreedyDecoder("layer_norm_lstm")

  def testBasicLayerNormLSTMSamplingDecoder(self):
    self._testLSTMLikeSamplingDecoder("layer_norm_lstm")

  def testBasicLayerNormLSTMBeamDecoder(self):
    self._testLSTMLikeBeamDecoder("layer_norm_lstm")

  def testBasicNASGreedyDecoder(self):
    self._testLSTMLikeGreedyDecoder("nas")

  def testBasicNASSamplingDecoder(self):
    self._testLSTMLikeSamplingDecoder("nas")

  def testBasicNASBeamDecoder(self):
    self._testLSTMLikeBeamDecoder("nas")

  def testBasicGRUGreedyDecoder(self):
    pass

  def testBasicGRUSamplingDecoder(self):
    pass

  def testBasicGRUBeamDecoder(self):
    pass


if __name__ == "__main__":
  tf.test.main()
