# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""
A direct copy of the object_detection package in the tensorflow/models repository.

The following commands were used to create this package (on 6 Nov 2019):

```
git clone https://github.com/tensorflow/models
cd models/research
git reset --hard b9ef963d1e84da0bb9c0a6039457c39a699ea149
cd object_detection
rm -r data dockerfiles g3doc test_ckpt test_data test_images CONTRIBUTING.md object_detection_tutorial.ipynb README.md
cd ..
cp -r object_detection $ROOT_DIR/tfrecords/src/wai/tfrecords/object_detection
cp ../LICENSE $ROOT_DIR/tfrecords/src/wai/tfrecords/object_detection
```

Modifications to source:
- Changed ``from object_detection import ...`` to ``from wai.tfrecords.object_detection import ...`` to match new
  namespace.
"""
