import os
from typing import Dict, Optional, List, Callable

import tensorflow as tf
import contextlib2
from wai.common.file.report import Report, loadf
from wai.common.file.report.constants import EXTENSION as REPORT_EXT
from wai.common.adams.imaging.locateobjects import LocatedObjects
from object_detection.dataset_tools import tf_record_creation_util

from ._logging import logger
from ._determine_labels import determine_labels
from ._get_files_from_directory import get_files_from_directory
from ._write_protobuf_label_map import write_protobuf_label_map
from ._ImageFormat import ImageFormat
from .constants import PREFIX_OBJECT
from ._fix_labels import fix_labels
from ._to_tf_example import to_tf_example


def convert(input_dir: Optional[str],
            input_files: Optional[List[str]],
            output_file: str,
            mappings: Optional[Dict[str, str]] = None,
            regexp: str = None,
            labels: Optional[List[str]] = None,
            protobuf_label_map: Optional[str] = None,
            shards: int = -1,
            verbose: bool = False):
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
    # Determine the list of files to convert
    if input_dir is not None:
        report_files = get_files_from_directory(input_dir)
    else:
        report_files = [os.path.splitext(input_file)[0] + REPORT_EXT for input_file in input_files]

    # Logging
    if verbose:
        logger.info(f"# report files: {len(report_files)}")

    # Determine the labels if they are not given
    if labels is None:
        labels = determine_labels(report_files, mappings, regexp, verbose)

    # Create a map from label to its index
    label_index_map: Dict[str, int] = {label: index + 1 for index, label in enumerate(labels)}

    # Output the label index map if requested
    if protobuf_label_map is not None:
        write_protobuf_label_map(label_index_map, protobuf_label_map)

    # Logging
    if verbose:
        logger.info(f"labels considered: {labels}")

    if shards > 1:
        with contextlib2.ExitStack() as tf_record_close_stack:
            output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(tf_record_close_stack, output_file, shards)

            class Writer:
                def __init__(self):
                    self.index = 0

                def __call__(self, example: tf.train.Example):
                    output_tfrecords[self.index % shards].write(example.SerializeToString())
                    self.index += 1

            do_convert(report_files, mappings, label_index_map, verbose, Writer())

    else:
        writer = tf.python_io.TFRecordWriter(output_file)

        def write(example: tf.train.Example):
            writer.write(example.SerializeToString())

        do_convert(report_files, mappings, label_index_map, verbose, write)

        writer.close()


def do_convert(report_files: List[str],
               mappings: Dict[str, str],
               label_index_map: Dict[str, int],
               verbose: bool,
               write: Callable[[tf.train.Example], None]):
    for report_file in report_files:
        image_file, image_format = ImageFormat.get_associated_image(report_file)

        if image_file is None:
            logger.warning(f"Failed to determine image for report: {report_file}")
            continue

        report: Report = loadf(report_file)

        objects: LocatedObjects = LocatedObjects.from_report(report, PREFIX_OBJECT)

        if mappings is not None:
            fix_labels(objects, mappings)

        if len(objects) > 0:
            example = to_tf_example(image_file, image_format, objects, label_index_map, verbose)

            if example is None:
                continue

            logger.info(f"storing: {image_file}")

            write(example)