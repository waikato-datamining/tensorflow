from typing import Dict

from .constants import SUFFIX_TYPE

from wai.common.adams.imaging.locateobjects import LocatedObjects


def fix_labels(objects: LocatedObjects, mappings: Dict[str, str]):
    """
    Fixes the labels in the parsed objects, using the specified mappings (old: new).

    :param objects: the parsed objects
    :type objects: dict
    :param mappings: the label mappings (old: new)
    :type mappings: dict
    """
    for object in objects:
        if SUFFIX_TYPE in object.metadata:
            label: str = object.metadata[SUFFIX_TYPE]
            if label in mappings:
                object.metadata[SUFFIX_TYPE] = mappings[label]
