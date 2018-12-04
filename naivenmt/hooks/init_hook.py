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


class InitHook(tf.train.SessionRunHook):

  def after_create_session(self, session, coord):
    iterator_init_op = tf.get_collection(collection_utils.ITERATOR)
    tables_init_op = tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS)
    variables_init_op = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    session.run(iterator_init_op)
    session.run(tables_init_op)
    session.run(variables_init_op)
