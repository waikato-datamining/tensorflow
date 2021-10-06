import json
import numpy as np
from datetime import datetime

from wai.tmm.common.predict_utils import set_input_tensor, get_output_tensor


class ClassificationResults(object):
    """
    Container class for storing classification results.
    """

    def __init__(self, timestamp, predictions, meta=None):
        """
        Initializes the container.

        :param timestamp: the timestamp string
        :type timestamp: str
        :param predictions: the dictionary of classifications (label -> probability)
        :type predictions: dict
        :param meta: the optional dictionary with meta data
        :type meta: dict
        """
        self.timestamp = timestamp
        self.predictions = predictions
        self.meta = meta

    def to_json_string(self, indent=None):
        """
        Returns a json string.

        :param indent: the indentation, use None for minified string
        :type indent: int
        :return: the generated json string
        :rtype: str
        """
        d = {
            "timestamp": self.timestamp,
            "predictions": self.predictions,
        }
        if self.meta is not None:
            d["meta"] = self.meta
        if indent is None:
            return json.dumps(d)
        else:
            return json.dumps(d, indent=indent)


def classify_image(interpreter, image, threshold=0.3, labels=None, mean=0.0, stdev=255.0):
    """
    Returns a list of detection results, each a dictionary of object info.

    :param interpreter: the model to use use
    :param image: the preprocessed image to make a prediction for
    :type image: np.ndarray
    :param threshold: the probability threshold to use
    :type threshold: float
    :param labels: the class labels
    :type labels: list
    :param mean: the mean to use for the input image
    :type mean: float
    :param stdev: the stdev to use for the input image
    :type stdev: float
    :return: the predictions
    :rtype: ClassificationResults
    """
    start = datetime.now()

    floating_model = interpreter.get_input_details()[0]['dtype'] == np.float32
    input_data = np.expand_dims(image, axis=0)
    if floating_model:
        input_data = (np.float32(input_data) - mean) / stdev

    # Feed the input image to the model
    set_input_tensor(interpreter, input_data)
    interpreter.invoke()

    end = datetime.now()
    meta = {"prediction_time": str((end-start).total_seconds())}

    # make predictions
    classifications = get_output_tensor(interpreter, 0)
    preds = {}
    for i in range(len(classifications)):
        if classifications[i] > threshold:
            preds[labels[i]] = float(classifications[i])

    return ClassificationResults(str(start), preds, meta=meta)
