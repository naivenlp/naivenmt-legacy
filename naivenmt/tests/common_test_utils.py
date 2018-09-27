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

import argparse
import os

from naivenmt.configs import add_arguments


def get_testdata_dir():
  return os.path.abspath(os.path.join(os.pardir, "../", "testdata"))


def get_testdata_file(filename):
  return os.path.join(get_testdata_dir(), filename)


def add_required_params(flags):
  flags.out_dir = "/tmp/model"
  flags.src = "en"
  flags.tgt = "vi"
  flags.train_prefix = os.path.join(get_testdata_dir(), "iwslt15.tst2013.100")
  flags.dev_prefix = os.path.join(get_testdata_dir(), "iwslt15.tst2013.100")
  flags.test_prefix = os.path.join(get_testdata_dir(), "iwslt15.tst2013.100")
  flags.vocab_prefix = os.path.join(get_testdata_dir(), "iwslt15.vocab.100")


def setup_flags():
  parser = argparse.ArgumentParser()
  add_arguments(parser)
  flags, _ = parser.parse_known_args()
  add_required_params(flags)
  return flags
