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


class BasicDecodersTrainOrEvalTest(tf.test.TestCase):

  def runGRUDecoder(self,
                    decoder,
                    num_layers,
                    encoder_outputs,
                    encoder_states,
                    labels,
                    src_seq_len):
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
        self.assertEqual(tf.int32, predict_ids.dtype)
        # TODO(luozhouyang) why logits.shape[0] is not certain? 0/1/2
        self.assertEqual(logits.shape[0], predict_ids.shape[0])

        # final states is a tuple of (states_c, states_h) of length num_layers
        self.assertEqual(num_layers, len(states))
        for i in range(num_layers):
          self.assertAllEqual([utils.BATCH_SIZE, utils.DEPTH], states[i].shape)

  def runLSTMDecoder(self,
                     decoder,
                     num_layers,
                     encoder_outputs,
                     encoder_states,
                     labels,
                     src_seq_len):
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

  def testBasicLSTMDecoder(self):
    for num_layers in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
      configs = {
        "num_encoder_layers": num_layers,
        "num_decoder_layers": num_layers,
        "encoder_type": "uni",
        "unit_type": "lstm",
        "time_major": True
      }
      decoder = utils.build_basic_decoder(configs)
      # target_vocab_size if 6, which determines the shape of logits
      self.assertEqual(6, decoder.target_vocab_size)
      enc_outputs, enc_outputs_len, enc_states = (
        utils.get_uni_lstm_encoder_results(num_layers))
      self.runLSTMDecoder(decoder,
                          num_layers,
                          enc_outputs,
                          enc_states,
                          utils.get_labels(),
                          enc_outputs_len)

  def testBasicGRUDecoder(self):
    for num_layers in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
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
      self.runGRUDecoder(decoder,
                         num_layers,
                         enc_outputs,
                         enc_states,
                         utils.get_labels(),
                         enc_outputs_len)

  def testBasicLayerNormLSTMDecoder(self):
    for num_layers in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
      configs = {
        "num_encoder_layers": num_layers,
        "num_decoder_layers": num_layers,
        "encoder_type": "uni",
        "unit_type": "layer_norm_lstm",
        "time_major": True
      }
      decoder = utils.build_basic_decoder(configs)
      # target_vocab_size if 6, which determines the shape of logits
      self.assertEqual(6, decoder.target_vocab_size)
      enc_outputs, enc_outputs_len, enc_states = (
        utils.get_uni_layer_norm_lstm_encoder_results(num_layers))
      self.runLSTMDecoder(decoder,
                          num_layers,
                          enc_outputs,
                          enc_states,
                          utils.get_labels(),
                          enc_outputs_len)

  def testBasicNASDecoder(self):
    for num_layers in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
      configs = {
        "num_encoder_layers": num_layers,
        "num_decoder_layers": num_layers,
        "encoder_type": "uni",
        "unit_type": "nas",
        "time_major": True
      }
      decoder = utils.build_basic_decoder(configs)
      # target_vocab_size if 6, which determines the shape of logits
      self.assertEqual(6, decoder.target_vocab_size)
      enc_outputs, enc_outputs_len, enc_states = (
        utils.get_uni_nas_encoder_results(num_layers))
      self.runLSTMDecoder(decoder,
                          num_layers,
                          enc_outputs,
                          enc_states,
                          utils.get_labels(),
                          enc_outputs_len)


class BasicDecodersPredictTest(tf.test.TestCase):

  def runLSTMGreedyDecoder(self, decoder, num_layers):
    pass

  def runLSTMBeamDecoder(self, decoder, num_layers):
    pass

  def runGRUGreedyDecoder(self, decoder, num_layers):
    pass

  def runGRUBeamDecoder(self, decoder, num_layers):
    pass

  def testBasicLSTMGreedyDecoder(self):
    pass

  def testBasicLSTMBeamDecoder(self):
    pass

  def testBasicLayerNormLSTMGreedyDecoder(self):
    pass

  def testBasicLayerNormLSTMBeamDecoder(self):
    pass

  def testBasicNASGreedyDecoder(self):
    pass

  def testBasicNASBeamDecoder(self):
    pass

  def testBasicGRUGreedyDecoder(self):
    pass

  def testBasicGRUBeamDecoder(self):
    pass


if __name__ == "__main__":
  tf.test.main()
