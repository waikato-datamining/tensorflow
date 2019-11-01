import re
from typing import Optional, Dict, Pattern, Set, List

from wai.common.adams.imaging.locateobjects import LocatedObjects
from wai.common.file.report import loadf, Report

from ._fix_labels import fix_labels
from .constants import SUFFIX_TYPE, DEFAULT_LABEL, PREFIX_OBJECT


def determine_labels(input_files: List[str],
                     mappings: Optional[Dict[str, str]] = None,
                     regexp: Optional[str] = None):
    """
    Determines all the labels present in the reports and returns them.
    The labels get updated using the mappings before the label regexp is tested.

    :param input_files:     The list of report files to read labels from.
    :param mappings:        The label mappings for replacing labels (key: old label, value: new label).
    :param regexp:          The regular expression to use for limiting the labels stored.
    :return:                The list of labels.
    """
    # Compile the regex if given
    regexpc: Pattern = re.compile(regexp) if regexp is not None else None

    labels: Set[str] = set()
    for report_file in input_files:
        # Load the report
        report: Report = loadf(report_file)

        # Read any objects from the report
        objects: LocatedObjects = LocatedObjects.from_report(report, PREFIX_OBJECT)

        # Map the labels if a mapping is given
        if mappings is not None:
            fix_labels(objects, mappings)

        # Get the label from each object
        for obj in objects:
            # Get the label
            label: str = obj.metadata[SUFFIX_TYPE] if SUFFIX_TYPE in obj.metadata else DEFAULT_LABEL

            # Add the label to the set (if allowed)
            if regexpc is None or regexpc.match(label):
                labels.add(label)

    # create sorted list
    result = list(labels)
    result.sort()

    return result
