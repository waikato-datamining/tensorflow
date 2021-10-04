import argparse
import traceback

from wai.tmm.objdet.predict_utils import load_model, load_classes, preprocess_image, detect_objects


def predict(model, labels, image, threshold, output=None):
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
    """

    interpreter = load_model(model)
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    classes = load_classes(labels)

    preprocessed_image, original_image = preprocess_image(image, (input_height, input_width))
    image_height, image_width, _ = original_image.shape
    results = detect_objects(interpreter, preprocessed_image, (image_height, image_width), threshold=threshold, labels=classes)
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
        description="Uses an object detection model to make predictions on a single image.",
        prog="tmm-od-predict",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', metavar="FILE", type=str, required=True, help='The tflite object detection model to use.')
    parser.add_argument('--labels', metavar="FILE", type=str, required=True, help='The text file with the labels (one label per line).')
    parser.add_argument('--image', metavar="FILE", type=str, required=True, help='The image to make the prediction for.')
    parser.add_argument('--threshold', metavar="0-1", type=float, required=False, default=0.3, help='The probability threshold to use.')
    parser.add_argument('--output', metavar="FILE", type=str, required=False, help='The JSON file to store the predictions in.')
    parsed = parser.parse_args(args=args)

    predict(parsed.model, parsed.labels, parsed.image, parsed.threshold, parsed.output)


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
