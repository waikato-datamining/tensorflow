import argparse
import traceback
import numpy as np

from wai.tmm.common.predict_utils import preprocess_image
from wai.tmm.common.io import load_model, load_classes
from wai.tmm.imgcls.predict_utils import classify_image


def predict(model, labels, image, threshold, output=None, mean=0.0, stdev=255.0):
    """
    Uses an object detection model to make a prediction for a single image.

    :param model: the model to load
    :type model: str
    :param labels: the text file with the labels (one label per line)
    :type labels: str
    :param image: the image to make the predictions for
    :type image: str
    :param threshold: the probability threshold to use
    :type threshold: float
    :param output: the JSON file to store the predictions in, gets output to stdout if None
    :type output: str
    :param mean: the mean to use for the input image
    :type mean: float
    :param stdev: the stdev to use for the input image
    :type stdev: float
    """

    interpreter = load_model(model)
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    classes = load_classes(labels)

    preprocessed_image, _ = preprocess_image(image, (input_height, input_width))
    results = classify_image(interpreter, preprocessed_image, threshold=threshold, labels=classes, mean=mean, stdev=stdev)

    if output is None:
        print(results.to_json_string())
    else:
        with open(output, "w") as f:
            f.write(results.to_json_string())


def main(args=None):
    """
    The main method for parsing command-line arguments and starting the training.

    :param args: the commandline arguments, uses sys.argv if not supplied
    :type args: list
    """

    parser = argparse.ArgumentParser(
        description="Uses a tflite image classification model to make predictions on a single image.",
        prog="tmm-ic-predict",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', metavar="FILE", type=str, required=True, help='The tflite object detection model to use.')
    parser.add_argument('--labels', metavar="FILE", type=str, required=True, help='The text file with the labels (one label per line).')
    parser.add_argument('--image', metavar="FILE", type=str, required=True, help='The image to make the prediction for.')
    parser.add_argument('--threshold', metavar="0-1", type=float, required=False, default=0.3, help='The probability threshold to use.')
    parser.add_argument('--output', metavar="FILE", type=str, required=False, help='The JSON file to store the predictions in, prints to stdout if omitted.')
    parser.add_argument('--mean', metavar="NUM", type=float, required=False, default=0.0, help='The mean to use for the input image.')
    parser.add_argument('--stdev', metavar="NUM", type=float, required=False, default=255.0, help='The stdev to use for the input image.')
    parsed = parser.parse_args(args=args)

    predict(parsed.model, parsed.labels, parsed.image, parsed.threshold, output=parsed.output,
            mean=parsed.mean, stdev=parsed.stdev)


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


if __name__ == '__main__':
    main()
