"""
Copyright 2018-2019 University of Waikato, Hamilton, NZ

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

import sys
import argparse
import os
import io
import traceback
import logging
import numpy as np
import PIL.Image as pil
import tensorflow as tf
import contextlib2
from object_detection.utils import dataset_util
from object_detection.dataset_tools import tf_record_creation_util
from adams2objectdetection.report import read_objects, determine_labels, fix_labels
from adams2objectdetection.report import SUFFIX_TYPE, SUFFIX_X, SUFFIX_Y, SUFFIX_WIDTH, SUFFIX_HEIGHT, REPORT_EXT
from adams2objectdetection.report import PREFIX_OBJECT, DEFAULT_LABEL

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
    filename = (os.path.basename(imgpath)).encode('utf8')

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    for o in objects.values():
        if SUFFIX_TYPE in o:
            if o[SUFFIX_TYPE] in labels:
                if (o[SUFFIX_X] < 0) or (o[SUFFIX_Y] < 0) or (o[SUFFIX_WIDTH] < 0) or (o[SUFFIX_HEIGHT] < 0):
                    continue
                x0 = o[SUFFIX_X] / width
                x1 = (o[SUFFIX_X] + o[SUFFIX_WIDTH] - 1) / width
                y0 = o[SUFFIX_Y] / height
                y1 = (o[SUFFIX_Y] + o[SUFFIX_HEIGHT] - 1) / height
                if ((x0 >= 0) and (x0 <= 1.0) and (x1 >= 0) and (x1 <= 1.0) and (x0 < x1)) \
                        and ((y0 >= 0) and (y0 <= 1.0) and (y1 >= 0) and (y1 <= 1.0) and (y0 < y1)):
                    xmins.append(x0)
                    xmaxs.append(x1)
                    ymins.append(y0)
                    ymaxs.append(y1)
                    classes_text.append(o[SUFFIX_TYPE].encode('utf8'))
                    classes.append(labels[o[SUFFIX_TYPE]])
    if len(xmins) == 0:
        logger.warning("No annotations in '" + str(imgpath) + "', skipping!")
        return None

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


def determine_image(report):
    """
    Determines the image and image type that is associated with the report file.

    :param report: the report file to get the image/type for
    :type report: str
    :return: image and image type (str and bytearray)
    :rtype: tuple
    """

    jpgLower = report.replace(REPORT_EXT, ".jpg")
    jpgUpper = report.replace(REPORT_EXT, ".JPG")
    pngLower = report.replace(REPORT_EXT, ".png")
    pngUpper = report.replace(REPORT_EXT, ".PNG")
    img = None
    imgtype = None

    if os.path.exists(jpgLower):
        img = jpgLower
        imgtype = b'jpg'
    elif os.path.exists(jpgUpper):
        img = jpgUpper
        imgtype = b'jpg'
    elif os.path.exists(pngLower):
        img = pngLower
        imgtype = b'png'
    elif os.path.exists(pngUpper):
        img = pngUpper
        imgtype = b'png'

    return img, imgtype


def convert(input_dir, input_files, output_file, mappings=None, regexp=None, labels=None, protobuf_label_map=None,
            shards=-1, verbose=False):
    """
    Converts the images and annotations (.report) files into TFRecords.

    :param input_dir: the input directory (PNG/JPG, .report)
    :type input_dir: str
    :param input_files: the file containing the report files to use
    :type input_files: str
    :param output_file: the output file for TFRecords
    :type output_file: str
    :param mappings: the label mappings for replacing labels (key: old label, value: new label)
    :type mappings: dict
    :param regexp: the regular expression to use for limiting the labels stored
    :type regexp: str
    :param labels: the predefined list of labels to use
    :type labels: list
    :param protobuf_label_map: the (optional) file to store the label mapping (in protobuf format)
    :type protobuf_label_map: str
    :param shards: the number of shards to generate, <= 1 for just single file
    :type shards: int
    :param verbose: whether to have a more verbose record generation
    :type verbose: bool
    """

    if labels is None:
        labels = determine_labels(input_dir=input_dir, input_files=input_files, mappings=mappings,
                                  regexp=regexp, verbose=verbose)
    label_indices = dict()
    for i, l in enumerate(labels):
        label_indices[l] = i+1

    if protobuf_label_map is not None:
        protobuf = list()
        for l in label_indices:
            protobuf.append("item {\n  id: %d\n  name: '%s'\n}\n" % (label_indices[l], l))
        with open(protobuf_label_map, 'w') as f:
            f.writelines(protobuf)

    if verbose:
        logger.info("labels considered: %s", labels)

    # determine files
    if input_dir is not None:
        report_files = list()
        for subdir, dirs, files in os.walk(input_dir):
            for f in files:
                if f.endswith(REPORT_EXT):
                    report_files.append(os.path.join(input_dir, subdir, f))
    else:
        report_files = list()
        for input_file in input_files:
            report_files.append(os.path.splitext(input_file)[0] + REPORT_EXT)

    if verbose:
        logger.info("# report files: %d", len(report_files))

    if shards > 1:
        index = 0
        with contextlib2.ExitStack() as tf_record_close_stack:
            output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
                tf_record_close_stack, output_file, shards)
            for report in report_files:
                img, imgtype = determine_image(report)
                if img is not None:
                    objects = read_objects(report, verbose=verbose)
                    if mappings is not None:
                        fix_labels(objects, mappings)
                    if len(objects) > 0:
                        example = create_record(img, imgtype, objects, label_indices, verbose)
                        if example is None:
                            continue
                        logger.info("storing: %s", img)
                        output_shard_index = index % shards
                        output_tfrecords[output_shard_index].write(example.SerializeToString())
                        index += 1
                else:
                    logger.warning("Failed to determine image for report: %s", report)
    else:
        writer = tf.python_io.TFRecordWriter(output_file)
        for report in report_files:
            img, imgtype = determine_image(report)
            if img is not None:
                objects = read_objects(report, verbose=verbose)
                if mappings is not None:
                    fix_labels(objects, mappings)
                if len(objects) > 0:
                    example = create_record(img, imgtype, objects, label_indices, verbose)
                    if example is None:
                        continue
                    logger.info("storing: %s", img)
                    writer.write(example.SerializeToString())
            else:
                logger.warning("Failed to determine image for report: %s", report)
        writer.close()


def main(args):
    """
    Runs the conversion from command-line. Use -h/--help to see all options.

    :param args: the command-line arguments to parse
    :type args: list
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
        "-p", "--protobuf_label_map", metavar="file", dest="protobuf_label_map", required=False,
        help="for storing the label strings and IDs", default=None)
    parser.add_argument(
        "-m", "--mapping", metavar="old=new", dest="mapping", action='append', type=str, required=False,
        help="mapping for labels, for replacing one label string with another (eg when fixing/collapsing labels)", default=list())
    parser.add_argument(
        "-r", "--regexp", metavar="regexp", dest="regexp", required=False,
        help="regular expression for using only a subset of labels", default="")
    parser.add_argument(
        "-l", "--labels", metavar="label1,label2,...", dest="labels", required=False,
        help="comma-separated list of labels to use", default="")
    parser.add_argument(
        "-s", "--shards", metavar="num", dest="shards", required=False, type=int,
        help="number of shards to split the images into (<= 1 for off)", default=-1)
    parser.add_argument(
        "-v", "--verbose", action="store_true", dest="verbose", required=False,
        help="whether to be more verbose when generating the records")
    parsed = parser.parse_args(args=args)

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

    # generate label mappings
    mappings = None
    if len(parsed.mapping) > 0:
        mappings = dict()
        for m in parsed.mapping:
            old, new = m.split("=")
            mappings[old] = new

    # predefined labels?
    labels = None
    if len(parsed.labels) > 0:
        labels = list(parsed.labels.split(","))
        logger.info("labels: " + str(labels))

    if parsed.verbose:
        logger.info("sharding off" if parsed.shards <= 1 else "# shards: " + str(parsed.shards))

    convert(
        input_dir=input_dir, input_files=input_files, output_file=parsed.output, regexp=parsed.regexp,
        shards=parsed.shards, mappings=mappings, labels=labels, protobuf_label_map=parsed.protobuf_label_map,
        verbose=parsed.verbose)

if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as ex:
        print(traceback.format_exc())
