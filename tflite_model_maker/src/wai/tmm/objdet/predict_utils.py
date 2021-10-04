import numpy as np
from datetime import datetime
from opex import ObjectPredictions, ObjectPrediction, BBox, Polygon

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)


def load_model(model):
    """
    Loads the model.

    :param model: the model to load
    :type model: str
    :return: the interpreter
    """
    interpreter = tf.lite.Interpreter(model_path=model)
    interpreter.allocate_tensors()
    return interpreter


def load_classes(labels):
    """
    Loads the labels from the text file (one label per line).

    :param labels: the file with the labels
    :type labels: str
    :return: the list of labels
    :rtype: list
    """
    with open(labels, "r") as f:
        classes = [x.strip() for x in f.readlines()]
    return classes


def preprocess_image(image_path, input_size):
    """
    Preprocess the input image to feed to the TFLite model.

    :param image_path: the image to load
    :type image_path: str
    :param input_size: the tuple (height, width) to resize to
    :type input_size: tuple
    :return: the preprocessed image
    :rtype: np.ndarray
    """
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    original_image = img
    resized_img = tf.image.resize(img, input_size)
    resized_img = resized_img[tf.newaxis, :]
    return resized_img, original_image


def preprocess_image_bytes(image_data, input_size):
    """
    Preprocess the input image to feed to the TFLite model.

    :param image_data: the image bytes to load
    :type image_data: bytes
    :param input_size: the tuple (height, width) to resize to
    :type input_size: tuple
    :return: the preprocessed image
    :rtype: np.ndarray
    """
    img = tf.io.decode_image(image_data, channels=3)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    original_image = img
    resized_img = tf.image.resize(img, input_size)
    resized_img = resized_img[tf.newaxis, :]
    return resized_img, original_image


def set_input_tensor(interpreter, image):
    """
    Set the input tensor.

    :param interpreter: the model to feed in the image
    :param image: the preprocessed image to feed in
    """
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
    """
    Return the output tensor at the given index.

    :param interpreter: the model to get the tensor from
    :param index: the index of the tensor
    :type index: int
    :return: the tensor
    :rtype: nd.ndarray
    """
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor


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
