"""
Copyright 2018 University of Waikato, Hamilton, NZ

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
import javaproperties
import os
import re
import io
import traceback
import numpy as np
import PIL.Image as pil
import tensorflow as tf
from object_detection.utils import dataset_util

def read_objects(report_file, labels=None, verbose=False):
    """
    Reads the report file and returns a list of all the objects (each object is a dictionary).

    :param report_file: the report file to read
    :type report_file: str
    :param labels: the regular expression to use for limiting the labels stored
    :type labels: str
    :param verbose: whether to have a more verbose record generation
    :type verbose: bool
    :return: the dictionary of objects that matched
    :rtype: dict
    """

    result = dict()

    with open(report_file, 'r') as rf:
        props = javaproperties.load(rf)
        for k in props.keys():
            if k.startswith("Object"):
                idx = k[len("Object")+1:]
                idx = idx[0:idx.index(".")]
                if idx not in result:
                    result[idx] = dict()
                subkey = k[len("Object." + idx) + 1:]
                if subkey.endswith("DataType"):
                    continue
                value = props[k]
                # try guess type
                if (value.lower() == "true") or (value.lower() == "false"):
                    result[idx][subkey] = bool(props[k])
                else:
                    try:
                        result[idx][subkey] = float(props[k])
                    except:
                        result[idx][subkey] = props[k]

    if verbose:
        print(report_file, result)

    return result


def determine_labels(input_dir, labels=None, verbose=False):
    """
    Determines all the labels present in the reports and returns them.

    :param input_dir: the input directory (PNG/JPG, .report)
    :type input_dir: str
    :param labels: the regular expression to use for limiting the labels stored
    :type labels: str
    :param verbose: whether to have a more verbose record generation
    :type verbose: bool
    :return: the list of labels
    :rtype: list
    """

    if labels is not None:
        labelsc = re.compile(labels)
    else:
        labelsc = None

    resultset = set()
    for subdir, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".report"):
                objects = read_objects(os.path.join(input_dir, subdir, f), labels=labels, verbose=verbose)
                for o in objects.values():
                    if "type" in o:
                        if labelsc is not None:
                            if labelsc.match(o["type"]):
                                resultset.add(o["type"])
                        else:
                            resultset.add(o["type"])

    # create sorted list
    result = list(resultset)
    result.sort()

    return result


def create_tf_example(imgpath, imgtype, objects, labels, verbose):
    """
    Creates a tf.Example proto from image.

    Based on:
    https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
    https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_kitti_tf_record.py

    :param imgpath: the path to the image
    :type imgpath: str
    :param imgtype: the image type (jpg/png)
    :type imgtype: bytearray
    :param objects: the associated objects
    :type objects: dict
    :param labels: lookup for the numeric label indices via their label
    :type labels: dict
    :param verbose: whether to be verbose when creating the example
    :type verbose: bool
    :return: the generated example
    :rtype: tf.Example
    """

    with tf.gfile.GFile(imgpath, 'rb') as fid:
        encoded_img = fid.read()
    encoded_img_io = io.BytesIO(encoded_img)
    image = pil.open(encoded_img_io)
    image = np.asarray(image)

    height = int(image.shape[0])
    width = int(image.shape[1])
    filename = os.path.basename(imgpath)

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    for o in objects.values():
        if 'type' in o:
            xmins.append(o['x'] / width)
            xmaxs.append((o['x'] + o['width'] - 1) / width)
            ymins.append(o['y'] / height)
            ymaxs.append((o['y'] + o['height'] - 1) / height)
            classes_text.append(o['type'].encode('utf8'))
            classes.append(labels[o['type']])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_img),
        'image/format': dataset_util.bytes_feature(imgtype),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def convert(input_dir, output_dir, remove_alpha=False, labels=None, shards=10, verbose=False):
    """
    Converts the images and annotations (.report) files into TFRecords.

    :param input_dir: the input directory (PNG/JPG, .report)
    :type input_dir: str
    :param output_dir: the output directory for TFRecords
    :type output_dir: str
    :param remove_alpha: whether to remove the alpha channel
    :type remove_alpha: bool
    :param labels: the regular expression to use for limiting the labels stored
    :type labels: str
    :param shards: the number of images per record file
    :type shards: int
    :param verbose: whether to have a more verbose record generation
    :type verbose: bool
    """

    all_labels = determine_labels(input_dir, labels=labels, verbose=verbose)
    all_indices = dict()
    for i, l in enumerate(all_labels):
        all_indices[l] = i

    if verbose:
        print("determined labels:", all_labels)

    for subdir, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".report"):
                report = os.path.join(input_dir, subdir, f)
                jpg = os.path.join(input_dir, subdir, f.replace(".report", ".jpg"))
                png = os.path.join(input_dir, subdir, f.replace(".report", ".png"))
                img = None
                imgtype = None
                if os.path.exists(jpg):
                    img = jpg
                    imgtype = b'jpg'
                elif os.path.exists(png):
                    img = png
                    imgtype = b'png'
                if img is not None:
                    objects = read_objects(report, labels=labels, verbose=verbose)
                    example = create_tf_example(img, imgtype, objects, all_indices, verbose)
                    if verbose:
                        print(example)
                    # TODO write to shards


def main():
    """
    Runs the conversion from command-line. Use -h/--help to see all options.
    """

    parser = argparse.ArgumentParser(
        description='Converts ADAMS annotations (image and .report files) into TFRecords.\n' +
                    'Assumes "Object." as prefix and ".type" for the label. If no ".type" present, ' +
                    'the generic label "object" will be used instead.')
    parser.add_argument(
        "-i", "--input", metavar="input", dest="input", required=True,
        help="input directory with ADAMS annotations")
    parser.add_argument(
        "-a", "--remove_alpha", action="store_true", dest="remove_alpha", required=False,
        help="whether to remove the alpha channel")
    parser.add_argument(
        "-o", "--output", metavar="ouput", dest="output", required=True,
        help="output directory for TFRecords")
    parser.add_argument(
        "-l", "--labels", metavar="labels", dest="labels", required=False,
        help="regular expression for using only a subset of labels", default="")
    parser.add_argument(
        "-s", "--shards", metavar="shards", dest="shards", type=int, required=False,
        help="the number of images to store in a single TFRecord file", default=10)
    parser.add_argument(
        "-v", "--verbose", action="store_true", dest="verbose", required=False,
        help="whether to be more verbose when generating the records")
    parsed = parser.parse_args()

    # checks
    if not os.path.exists(parsed.input):
        raise IOError("Input directory does not exist:", parsed.input)
    if not os.path.isdir(parsed.input):
        raise IOError("Input is not a directory:", parsed.input)
    if not os.path.exists(parsed.output):
        raise IOError("Output directory does not exist:", parsed.output)
    if not os.path.isdir(parsed.output):
        raise IOError("Output is not a directory:", parsed.output)

    convert(
        input_dir=parsed.input, output_dir=parsed.output,
        remove_alpha=parsed.remove_alpha,
        labels=parsed.labels, shards=parsed.shards, verbose=parsed.verbose)

if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        print(traceback.format_exc())
