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

import javaproperties
import os
import re

REPORT_EXT = ".report"
""" extension for report files. """

PREFIX_OBJECT = "Object."
""" the prefix for objects in the report. """

SUFFIX_TYPE = "type"
""" the suffix for the label of objects. """

SUFFIX_DATATYPE = "DataType"
""" the suffix for the data type of a property in the report. """

DEFAULT_LABEL = "object"
""" the default label to use if no 'type' present. """


def read_objects(report_file, verbose=False):
    """
    Reads the report file and returns a list of all the objects (each object is a dictionary).

    :param report_file: the report file to read
    :type report_file: str
    :param verbose: whether to have a more verbose record generation
    :type verbose: bool
    :return: the dictionary of objects that matched
    :rtype: dict
    """

    result = dict()

    with open(report_file, 'r') as rf:
        props = javaproperties.load(rf)
        for k in props.keys():
            if k.startswith(PREFIX_OBJECT):
                idx = k[len(PREFIX_OBJECT):]
                idx = idx[0:idx.index(".")]
                if idx not in result:
                    result[idx] = dict()
                subkey = k[len(PREFIX_OBJECT + idx) + 1:]
                if subkey.endswith(SUFFIX_DATATYPE):
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


def determine_labels(input_dir=None, input_files=None, labels=None, verbose=False):
    """
    Determines all the labels present in the reports and returns them.
    Can either locate report files recursively in a directory or directly use
    a list of report file names.

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

    # determine files
    if input_dir is not None:
        report_files = list()
        for subdir, dirs, files in os.walk(input_dir):
            for f in files:
                if f.endswith(REPORT_EXT):
                    report_files.append(os.path.join(input_dir, subdir, f))
    else:
        report_files = input_files[:]

    result_set = set()
    for report_file in report_files:
        objects = read_objects(report_file, verbose=verbose)
        for o in objects.values():
            if SUFFIX_TYPE in o:
                l = o[SUFFIX_TYPE]
            else:
                l = DEFAULT_LABEL
            if labelsc is not None:
                if labelsc.match(l):
                    result_set.add(l)
            else:
                result_set.add(l)

    # create sorted list
    result = list(result_set)
    result.sort()

    return result
