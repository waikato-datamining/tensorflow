import os
from typing import Dict, Optional, List, Callable

import tensorflow as tf
import contextlib2
from wai.common.file.report import Report, loadf
from wai.common.file.report.constants import EXTENSION as REPORT_EXT
from wai.common.adams.imaging.locateobjects import LocatedObjects

from ..object_detection.dataset_tools import tf_record_creation_util
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
    :param input_files: the file containing the report files to use
    :param output_file: the output file for TFRecords
    :param mappings: the label mappings for replacing labels (key: old label, value: new label)
    :param regexp: the regular expression to use for limiting the labels stored
    :param labels: the predefined list of labels to use
    :param protobuf_label_map: the (optional) file to store the label mapping (in protobuf format)
    :param shards: the number of shards to generate, <= 1 for just single file
    :param verbose: whether to have a more verbose record generation
    """
    # Determine the list of files to convert
    if input_dir is not None:
        report_files = get_files_from_directory(input_dir)
    else:
        report_files = [os.path.splitext(input_file)[0] + REPORT_EXT for input_file in input_files]

    # Log the report files
    if verbose:
        logger.info(f"# report files: {len(report_files)}")

    # Determine the labels if they are not given
    if labels is None:
        labels = determine_labels(report_files, mappings, regexp)

    # Create a map from label to its index
    label_index_map: Dict[str, int] = {label: index + 1 for index, label in enumerate(labels)}

    # Output the label index map if requested
    if protobuf_label_map is not None:
        write_protobuf_label_map(label_index_map, protobuf_label_map)

    # Log the labels
    if verbose:
        logger.info(f"labels considered: {labels}")

    if shards > 1:
        # Sharded output
        convert_sharded(report_files, output_file, mappings, label_index_map, verbose, shards)
    else:
        # Unsharded output
        convert_unsharded(report_files, output_file, mappings, label_index_map, verbose)


def convert_sharded(report_files: List[str],
                    output_file: str,
                    mappings: Optional[Dict[str, str]],
                    label_index_map: Dict[str, int],
                    verbose: bool,
                    shards: int):
    """
    Performs the conversion in a sharded manner.

    :param report_files:        The list of report files to convert.
    :param output_file:         The file to write the conversion results to.
    :param mappings:            Label mappings.
    :param label_index_map:     The label index lookup.
    :param verbose:             Whether to log verbose messages.
    :param shards:              The number of shards.
    """
    with contextlib2.ExitStack() as tf_record_close_stack:
        # Open the output file for sharded writing
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(tf_record_close_stack,
                                                                                 output_file,
                                                                                 shards)

        # Functor for sharded writing
        class Writer:
            def __init__(self):
                self.index = 0

            def __call__(self, example: tf.train.Example):
                output_tfrecords[self.index % shards].write(example.SerializeToString())
                self.index += 1

        # Perform the conversion
        do_convert(report_files, mappings, label_index_map, verbose, Writer())


def convert_unsharded(report_files: List[str],
                      output_file: str,
                      mappings: Optional[Dict[str, str]],
                      label_index_map: Dict[str, int],
                      verbose: bool):
    """
    Performs the conversion in an unsharded manner.

    :param report_files:        The list of report files to convert.
    :param output_file:         The file to write the conversion results to.
    :param mappings:            Label mappings.
    :param label_index_map:     The label index lookup.
    :param verbose:             Whether to log verbose messages.
    """
    # Create an unsharded writer
    writer = tf.io.TFRecordWriter(output_file)

    # Create a function to perform writing for do_convert
    def write(example: tf.train.Example):
        writer.write(example.SerializeToString())

    # Perform the conversion
    do_convert(report_files, mappings, label_index_map, verbose, write)

    # Close the writer
    writer.close()


def do_convert(report_files: List[str],
               mappings: Optional[Dict[str, str]],
               label_index_map: Dict[str, int],
               verbose: bool,
               write: Callable[[tf.train.Example], None]):
    """
    Performs the actual conversion of the report files, using the given
    write function to output the results.

    :param report_files:        The list of report files to convert.
    :param mappings:            Label mappings.
    :param label_index_map:     The label index lookup.
    :param verbose:             Whether to log verbose messages.
    :param write:               The function to call to output an example.
    """
    # Process each file
    for report_file in report_files:
        # Get the image associated to this report
        image_file, image_format = ImageFormat.get_associated_image(report_file)

        # Log a warning if the image wasn't found
        if image_file is None:
            logger.warning(f"Failed to determine image for report: {report_file}")
            continue

        # Load the report
        report: Report = loadf(report_file)

        # Get the annotated objects from the report
        objects: LocatedObjects = LocatedObjects.from_report(report, PREFIX_OBJECT)

        # Skip reports with no annotated objects
        if len(objects) == 0:
            continue

        # Apply any label mappings
        if mappings is not None:
            fix_labels(objects, mappings)

        # Create a Tensorflow example from the image and annotations
        example: tf.train.Example = to_tf_example(image_file, image_format, objects, label_index_map, verbose)

        # Continue if example-creation failed
        if example is None:
            continue

        # Logging
        logger.info(f"storing: {image_file}")

        # Output the example
        write(example)
