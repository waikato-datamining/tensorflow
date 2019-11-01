# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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


def logging_level_verbosity(logging_verbosity):
    """Converts logging_level into TensorFlow logging verbosity value

    Args:
      logging_level: String value representing logging level: 'DEBUG', 'INFO',
      'WARN', 'ERROR', 'FATAL'
    """
    name_to_level = {
        'FATAL': tf.compat.v1.logging.FATAL,
        'ERROR': tf.compat.v1.logging.ERROR,
        'WARN': tf.compat.v1.logging.WARN,
        'INFO': tf.compat.v1.logging.INFO,
        'DEBUG': tf.compat.v1.logging.DEBUG
    }

    try:
        return name_to_level[logging_verbosity]
    except Exception as e:
        raise RuntimeError('Not supported logs verbosity (%s). Use one of %s.' %
                           (str(e), list(name_to_level)))
