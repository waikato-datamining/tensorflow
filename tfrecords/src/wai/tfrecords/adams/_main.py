import argparse
import os
import traceback
from typing import Dict, Optional, List

from wai.tfrecords.adams import logger, convert
from wai.tfrecords.adams.constants import PREFIX_OBJECT, SUFFIX_TYPE, DEFAULT_LABEL


def main(args: Optional[List[str]] = None):
    """
    Runs the conversion from command-line. Use -h/--help to see all options.

    :param args:    The command-line arguments to parse, or None to
                    use the system args.
    """
    # Parse the arguments
    parsed = setup_parser().parse_args(args=args)

    # Check the input and output are valid
    if not os.path.exists(parsed.input):
        raise IOError("Input does not exist:", parsed.input)
    if os.path.isdir(parsed.output):
        raise IOError("Output is a directory:", parsed.output)

    # Interpret input (dir or file with report file names?)
    if os.path.isdir(parsed.input):
        input_dir = parsed.input
        input_files = None
    else:
        input_dir = None
        with open(parsed.input) as fp:
            input_files = [line.strip() for line in fp]

    # Generate label mappings
    mappings: Optional[Dict[str, str]] = None
    if len(parsed.mapping) > 0:
        mappings = dict()
        for m in parsed.mapping:
            old, new = m.split("=")
            mappings[old] = new

    # Extract the labels if given
    labels = None
    if len(parsed.labels) > 0:
        labels = list(parsed.labels.split(","))
        logger.info("labels: " + str(labels))

    # Logging
    if parsed.verbose:
        logger.info("sharding off" if parsed.shards <= 1 else "# shards: " + str(parsed.shards))

    # Convert using the specified arguments
    convert(
        input_dir=input_dir,
        input_files=input_files,
        output_file=parsed.output,
        regexp=parsed.regexp,
        shards=parsed.shards,
        mappings=mappings,
        labels=labels,
        protobuf_label_map=parsed.protobuf_label_map,
        verbose=parsed.verbose
    )


def sys_main() -> int:
    """
    Runs the main function using the system cli arguments, and
    returns a system error code.

    :return:    0 for success, 1 for failure.
    """
    try:
        main()
        return 0
    except Exception:
        print(traceback.format_exc())
        return 1


def setup_parser() -> argparse.ArgumentParser:
    """
    Sets up the argument parser for the main method.

    :return:    The argument parser.
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
        help="mapping for labels, for replacing one label string with another (eg when fixing/collapsing labels)",
        default=list())
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

    return parser
