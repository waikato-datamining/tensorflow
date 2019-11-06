from typing import Dict

from .constants import SUFFIX_TYPE

from wai.common.adams.imaging.locateobjects import LocatedObjects


def fix_labels(objects: LocatedObjects, mappings: Dict[str, str]):
    """
    Fixes the labels in the parsed objects, using the specified mappings (old: new).

    :param objects:     The parsed objects.
    :param mappings:    The label mappings (old: new).
    """
    # Process each object
    for obj in objects:
        # If the object doesn't have a label, skip it
        if SUFFIX_TYPE not in obj.metadata:
            continue

        # Get the object's current label
        label: str = obj.metadata[SUFFIX_TYPE]

        # If there is a mapping for this label, change it
        if label in mappings:
            obj.metadata[SUFFIX_TYPE] = mappings[label]
