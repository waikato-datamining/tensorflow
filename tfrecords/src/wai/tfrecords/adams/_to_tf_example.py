import io
import os
from typing import Dict

from PIL import Image as pil
import tensorflow as tf
import numpy as np
from wai.common.adams.imaging.locateobjects import LocatedObjects

from ..object_detection.utils import dataset_util
from ._logging import logger
from ._ImageFormat import ImageFormat
from .constants import SUFFIX_TYPE


def to_tf_example(imgpath: str,
                  imgtype: ImageFormat,
                  objects: LocatedObjects,
                  labels: Dict[str, int],
                  verbose: bool) -> tf.train.Example:
    """
    Creates a tf.Example proto from image.

    Based on:
    https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
    https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_kitti_tf_record.py

    :param imgpath:     The path to the image.
    :param imgtype:     The image type (jpg/png).
    :param objects:     The associated objects.
    :param labels:      Lookup for the numeric label indices via their label.
    :param verbose:     Whether to be verbose when creating the example.
    :return:            The generated example.
    """
    # Load the image as a Numpy array
    with tf.io.gfile.GFile(imgpath, 'rb') as fid:
        encoded_img = fid.read()
    encoded_img_io = io.BytesIO(encoded_img)
    image = pil.open(encoded_img_io)
    image = np.asarray(image)

    height = int(image.shape[0])
    width = int(image.shape[1])
    filename = (os.path.basename(imgpath)).encode('utf-8')

    # Format and extract the relevant annotation parameters
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    for o in objects:
        if SUFFIX_TYPE not in o.metadata:
            continue

        if o.metadata[SUFFIX_TYPE] not in labels:
            continue

        if (o.x < 0) or (o.y < 0) or (o.width < 0) or (o.height < 0):
            continue

        x0 = o.x / width
        x1 = (o.x + o.width - 1) / width
        y0 = o.y / height
        y1 = (o.y + o.height - 1) / height
        if ((x0 >= 0) and (x0 <= 1.0) and (x1 >= 0) and (x1 <= 1.0) and (x0 < x1)) \
                and ((y0 >= 0) and (y0 <= 1.0) and (y1 >= 0) and (y1 <= 1.0) and (y0 < y1)):
            xmins.append(x0)
            xmaxs.append(x1)
            ymins.append(y0)
            ymaxs.append(y1)
            classes_text.append(o.metadata[SUFFIX_TYPE].encode('utf8'))
            classes.append(labels[o.metadata[SUFFIX_TYPE]])

    # Skip this image if not annotations are found
    if len(xmins) == 0:
        logger.warning(f"No annotations in '{imgpath}', skipping!")
        return None

    # Logging
    if verbose:
        logger.info(imgpath)
        logger.info("xmins: %s", xmins)
        logger.info("xmaxs: %s", xmaxs)
        logger.info("ymins: %s", ymins)
        logger.info("ymaxs: %s", ymaxs)
        logger.info("classes_text: %s", classes_text)
        logger.info("classes: %s", classes)

    # Create the example
    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/height': dataset_util.int64_feature(height),
                'image/width': dataset_util.int64_feature(width),
                'image/filename': dataset_util.bytes_feature(filename),
                'image/source_id': dataset_util.bytes_feature(filename),
                'image/encoded': dataset_util.bytes_feature(encoded_img),
                'image/format': dataset_util.bytes_feature(imgtype.name.lower().encode("utf-8")),
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label': dataset_util.int64_list_feature(classes),
            }
        )
    )

    return tf_example
