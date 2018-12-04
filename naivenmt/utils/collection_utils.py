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

ITERATOR = "iterator"


def add_dict_to_collection(name, tensors_dict):
  keys = name + "_keys"
  values = name + "_values"
  for k, v in tensors_dict:
    tf.add_to_collection(keys, k)
    tf.add_to_collection(values, v)


def get_dict_from_collection(name):
  keys = name + "_keys"
  values = name + "_values"
  keys = tf.get_collection(keys)
  values = tf.get_collection(values)
  return dict(zip(keys, values))


def add_to_collection(name, tensor):
  tf.add_to_collection(name, tensor)


def get_from_collection(name):
  return tf.get_collection(name)
