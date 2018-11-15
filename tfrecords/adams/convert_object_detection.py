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
import os
import io
import traceback
import logging
import numpy as np
import PIL.Image as pil
import tensorflow as tf
from object_detection.utils import dataset_util
from report import read_objects, determine_labels
from report import SUFFIX_TYPE, SUFFIX_X, SUFFIX_Y, SUFFIX_WIDTH, SUFFIX_HEIGHT, REPORT_EXT
from report import PREFIX_OBJECT, DEFAULT_LABEL

# logging setup
logging.basicConfig()
logger = logging.getLogger("tfrecords.adams.convert_object_detection")
logger.setLevel(logging.INFO)


def create_record(imgpath, imgtype, objects, labels, verbose):
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
        if SUFFIX_TYPE in o:
            xmins.append(o[SUFFIX_X] / width)
            xmaxs.append((o[SUFFIX_X] + o[SUFFIX_WIDTH] - 1) / width)
            ymins.append(o[SUFFIX_Y] / height)
            ymaxs.append((o[SUFFIX_Y] + o[SUFFIX_HEIGHT] - 1) / height)
            classes_text.append(o[SUFFIX_TYPE].encode('utf8'))
            classes.append(labels[o[SUFFIX_TYPE]])
    if verbose:
        logger.info(imgpath)
        logger.info("xmins: %s", xmins)
        logger.info("xmaxs: %s", xmaxs)
        logger.info("ymins: %s", ymins)
        logger.info("ymaxs: %s", ymaxs)
        logger.info("classes_text: %s", classes_text)
        logger.info("classes: %s", classes)

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


def convert(input_dir, input_files, output_file, labels=None, verbose=False):
    """
    Converts the images and annotations (.report) files into TFRecords.

    :param input_dir: the input directory (PNG/JPG, .report)
    :type input_dir: str
    :param input_files: the file containing the report files to use
    :type input_files: str
    :param output_file: the output file for TFRecords
    :type output_file: str
    :param labels: the regular expression to use for limiting the labels stored
    :type labels: str
    :param verbose: whether to have a more verbose record generation
    :type verbose: bool
    """

    all_labels = determine_labels(input_dir=input_dir, input_files=input_files, labels=labels, verbose=verbose)
    all_indices = dict()
    for i, l in enumerate(all_labels):
        all_indices[l] = i

    if verbose:
        logging.info("determined labels: %s", all_labels)

    # determine files
    if input_dir is not None:
        report_files = list()
        for subdir, dirs, files in os.walk(input_dir):
            for f in files:
                if f.endswith(REPORT_EXT):
                    report_files.append(os.path.join(input_dir, subdir, f))
    else:
        report_files = input_files[:]

    writer = tf.python_io.TFRecordWriter(output_file)
    for report in report_files:
        jpg = report.replace(REPORT_EXT, ".jpg")
        png = report.replace(REPORT_EXT, ".png")
        img = None
        imgtype = None
        if os.path.exists(jpg):
            img = jpg
            imgtype = b'jpg'
        elif os.path.exists(png):
            img = png
            imgtype = b'png'
        if img is not None:
            objects = read_objects(report, verbose=verbose)
            if len(objects) > 0:
                if verbose:
                    logger.info("storing: %s", img)
                example = create_record(img, imgtype, objects, all_indices, verbose)
                writer.write(example.SerializeToString())
    writer.close()


def main():
    """
    Runs the conversion from command-line. Use -h/--help to see all options.
    """

    parser = argparse.ArgumentParser(
        description='Converts ADAMS annotations (image and .report files) into TFRecords for the '
                    + 'Object Detection framework.\n'
                    + 'Assumes "' + PREFIX_OBJECT + '" as prefix and "' + SUFFIX_TYPE + '" for the label. '
                    + 'If no "' + SUFFIX_TYPE + '" present, the generic label "' + DEFAULT_LABEL + '" will '
                    + 'be used instead.')
    parser.add_argument(
        "-i", "--input", metavar="dir_or_file", dest="input", required=True,
        help="input directory with report files or text file with one absolute report file name per line")
    parser.add_argument(
        "-o", "--output", metavar="file", dest="output", required=True,
        help="name of output file for TFRecords")
    parser.add_argument(
        "-l", "--labels", metavar="regexp", dest="labels", required=False,
        help="regular expression for using only a subset of labels", default="")
    parser.add_argument(
        "-v", "--verbose", action="store_true", dest="verbose", required=False,
        help="whether to be more verbose when generating the records")
    parsed = parser.parse_args()

    # checks
    if not os.path.exists(parsed.input):
        raise IOError("Input does not exist:", parsed.input)
    if os.path.isdir(parsed.output):
        raise IOError("Output is a directory:", parsed.output)

    # interpret input (dir or file with report file names?)
    if os.path.isdir(parsed.input):
        input_dir = parsed.input
        input_files = None
    else:
        input_dir = None
        input_files = list()
        with open(parsed.input) as fp:
            for line in fp:
                input_files.append(line.strip())

    convert(
        input_dir=input_dir, input_files=input_files, output_file=parsed.output,
        labels=parsed.labels, verbose=parsed.verbose)

if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        print(traceback.format_exc())
