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

import collections
import json
import re
import os
import tensorflow as tf


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


def load_image_list(image_list):
    """
    Loads the specified image list (in JSON format).

    :param image_list: the list with images
    :type image_list: str
    :return: the dictionary, each key represents a list of images with the key being the label
    :rtype: dict
    """

    with open(image_list, "r") as lf:
        result = json.load(lf)

    return result


def locate_sub_dirs(image_dir):
    """
    Locates the sub directories in the specified directory and generates a dictionary with the label
    generated from the sub directory associated with the corresponding path.

    :param image_dir: the directory to scan for sub dirs
    :type image_dir: str
    :return: the dictionary of label -> sub-dir relation
    :rtype: dict
    """

    result = collections.OrderedDict()
    sub_dirs = sorted(x[0] for x in tf.io.gfile.walk(image_dir))
    for sub_dir in sub_dirs:
        if sub_dir == image_dir:
            continue
        result[dir_to_label(os.path.basename(sub_dir))] = sub_dir
    return result


def locate_images(dir, strip_path=False):
    """
    Locates all PNG and JPG images in the specified directory and returns the list of filenames.

    :param dir: the directory to look for images
    :type dir: str
    :param strip_path: whether to strip the path
    :type strip_path: bool
    :return: the list of images
    :rtype: list
    """

    result = []
    extensions = sorted(set(os.path.normcase(ext)  # Smash case on Windows.
                            for ext in ['JPEG', 'JPG', 'jpeg', 'jpg', 'png']))
    if dir.endswith("/"):
        dir = dir[:-1]

    tf.compat.v1.logging.info("Looking for images in '" + os.path.basename(dir) + "'")
    for extension in extensions:
        file_glob = os.path.join(dir, '*.' + extension)
        result.extend(tf.io.gfile.glob(file_glob))

    if strip_path:
        for i in range(len(result)):
            result[i] = os.path.basename(result[i])

    return result
