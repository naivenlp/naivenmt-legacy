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

  def runLSTMDecoder(self,
                     decoder,
                     num_layers,
                     encoder_outputs,
                     encoder_states,
                     labels,
                     src_seq_len):
    logits, predict_ids, final_states = decoder.decode(
      mode=tf.estimator.ModeKeys.TRAIN,
      encoder_outputs=encoder_outputs,
      encoder_state=encoder_states,
      labels=labels,
      src_seq_len=src_seq_len)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      logits, predict_ids, final_states = sess.run(
        [logits, predict_ids, final_states]
      )
      print(logits)
      print(predict_ids)
      print(final_states)

  def testBasicLSTMDecoder(self):
    for num_layers in [NUM_LAYERS_2, NUM_LAYERS_4]:
      configs = {
        "num_decoder_layers": num_layers,
        "unit_type": "lstm",
      }
      decoder = utils.build_basic_decoder(configs)
      enc_outputs, enc_outputs_len, enc_states = (
        utils.get_uni_lstm_encoder_results(num_layers))
      self.runLSTMDecoder(decoder,
                          num_layers,
                          enc_outputs,
                          enc_states,
                          utils.get_labels(),
                          enc_outputs_len)

  def testBasicGRUDecoder(self):
    pass

  def testBasicLayerNormLSTMDecoder(self):
    for num_layers in [NUM_LAYERS_2, NUM_LAYERS_4]:
      configs = {
        "num_decoder_layers": num_layers,
        "unit_type": "layer_norm_lstm",
      }
      decoder = utils.build_basic_decoder(configs)
      enc_outputs, enc_outputs_len, enc_states = (
        utils.get_uni_layer_norm_lstm_encoder_results(num_layers))
      self.runLSTMDecoder(decoder,
                          num_layers,
                          enc_outputs,
                          enc_states,
                          utils.get_labels(),
                          enc_outputs_len)

  def testBasicNASDecoder(self):
    for num_layers in [NUM_LAYERS_2, NUM_LAYERS_4]:
      configs = {
        "num_decoder_layers": num_layers,
        "unit_type": "nas",
      }
      decoder = utils.build_basic_decoder(configs)
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
