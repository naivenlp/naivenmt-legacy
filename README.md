# naivenmt
An experimental implementation of NMT using tensorflow.

Actually, this project is a refactor of [tensorflow/nmt](https://github.com/tensorflow/nmt), with some 
improvements:  

* Better code organization
* A really simple API to export the pre-trained model
* A **Transformer** model(TODO)

See more about the `GNMT`'s architecture and details, to [tensorflow/nmt's README](https://github.com/tensorflow/nmt)

This project is under development, but it will not take too long to it's first release.
**All kinds of contributions are welcome.**


- [TODO List](#TODO)
- [Usage](#Usage)
    - [Train](#Train)
    - [Evaluate](#Evaluate)
    - [Predict](#Predict)
    - [Export](#Export)


## TODO
* Add hooks to monitor training, evaluation and inference
* Add tests
* Implement **Transformer** model. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## Usage
Almost all the params are the same as [tensorflow/nmt](https://github.com/tensorflow/nmt) except `--mode`. It's easy to use for those who have an
experience for `tensroflow/nmt`.

For new users, the usage is simple, too.
Here are some examples.

### Train

```bash
python3 -m naivenmt.naivenmt \
    --mode=train \
    --src=en --tgt=vi \
    --out_dir=/tmp/model \
    --train_prefix=$YOUR_DATA_DIR/train \
    --dev_prefix=$YOUR_DATA_DIR/dev \
    --test_prefix=$YOUT_DATA_DIR/test \
    --vocab_prefix=$YOUR_DATA_DIR/vocab \
    --num_train_steps=10000 \
    ...(other params)

```
See the default values for hparams, to [arguments.py](naivenmt/configs/arguments.py)

### Evaluate

### Predict

### Export


## License
```bash
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

```
