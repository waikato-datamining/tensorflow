# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
# Copyright 2019 University of Waikato, Hamilton, NZ.
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

import json
import re
import os


def dir_to_label(dir_name):
    """
    Turns the directory name into a label (only alphanumerc and lowercase).

    :param dir_name: the directory to convert
    :type dir_name: str
    :return: the label name
    :rtype: str
    """

    return re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())


def save_image_list(image_lists, image_lists_dir):
    """
    Saves the image lists as JSON files in the specified directory as separate
    files (training, testing, validation).

    :param image_lists: the dictionary with the lists.
    :type image_lists: dict
    :param image_lists_dir: the directory to store the lists in
    :type image_lists_dir: str
    """

    for list_name in ["training", "testing", "validation"]:
        with open(os.path.join(image_lists_dir, list_name + ".json"), "w") as lf:
            file_names = {}
            for dirname in image_lists:
                file_names[dirname] = image_lists[dirname][list_name][:]
            json.dump(file_names, lf, sort_keys=True, indent=4)
