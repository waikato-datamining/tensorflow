from datetime import datetime
from opex import ObjectPredictions, ObjectPrediction, BBox, Polygon

import tensorflow as tf

from wai.tmm.common.predict_utils import set_input_tensor, get_output_tensor

assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)


def box_to_bbox_polygon(box, input_size):
    """
    Turns the normalized box into a BBox and Polygon.

    :param box: the box to convert
    :param input_size: the height, width tuple
    :type input_size: tuple
    :return: tuple of BBox and Polygon
    :rtype: tuple
    """
    height, width = input_size
    ymin, xmin, ymax, xmax = box
    xmin = int(xmin * width)
    xmax = int(xmax * width)
    ymin = int(ymin * height)
    ymax = int(ymax * height)
    bbox = BBox(left=xmin, top=ymin, right=xmax, bottom=ymax)
    poly = Polygon(points=[(xmin, ymin), (xmax,ymin), (xmax, ymax), (xmin, ymax)])
    return bbox, poly


def detect_objects(interpreter, image, image_size, threshold=0.3, labels=None):
    """
    Returns a list of detection results, each a dictionary of object info.

    :param interpreter: the model to use use
    :param image: the preprocessed image to make a prediction for
    :type image: np.ndarray
    :param image_size: the image size tuple (height, width)
    :type image_size: tuple
    :param threshold: the probability threshold to use
    :type threshold: float
    :param labels: the class labels
    :type labels: list
    :return: the predicted objects
    :rtype: ObjectPredictions
    """

    start = datetime.now()
    timestamp = str(start)

    # Feed the input image to the model
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    end = datetime.now()
    meta = {"prediction_time": str((end-start).total_seconds())}

    # Get all outputs from the model
    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))

    objs = []
    for i in range(count):
        if scores[i] >= threshold:
            label = str(classes[i])
            if labels is not None:
                label = labels[int(classes[i])]
            bbox, poly = box_to_bbox_polygon(boxes[i], image_size)
            obj = ObjectPrediction(score=float(scores[i]), label=label, bbox=bbox, polygon=poly)
            objs.append(obj)

    result = ObjectPredictions(timestamp=timestamp, id="", objects=objs, meta=meta)
    return result
