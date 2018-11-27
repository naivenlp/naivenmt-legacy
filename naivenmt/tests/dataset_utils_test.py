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

from naivenmt.configs import HParamsBuilder
from naivenmt.utils import dataset_utils


class DatasetUtilsTest(tf.test.TestCase):

  @staticmethod
  def getDatasetRequiredParams():
    return {
      "random_seed": 1000,
      "num_buckets": 5,
      "src_max_len": 50,
      "tgt_max_len": 50,
      "num_parallel_calls": 4,
      "buff_size": 1024,
      "skip_count": 0,
      "batch_size": 4
    }

  def testBuildTrainingDataset(self):
    hparams = HParamsBuilder(self.getDatasetRequiredParams()).build()
    features, labels = dataset_utils.build_dataset(
      hparams, tf.estimator.ModeKeys.TRAIN)
    with self.test_session() as sess:
      for _ in range(5):
        print(sess.run(features['inputs']))
        print(sess.run(labels['tgt_in']))
        print(sess.run(labels['tgt_out']))
        print()


if __name__ == "__main__":
  tf.test.main()
