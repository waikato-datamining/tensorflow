import argparse
import json
import os
import traceback
from PIL import Image
from image_complete import auto
from sfp import Poller
from predict_utils import load_labels, load_model, predict_image


SUPPORTED_EXTS = [".jpg", ".jpeg", ".png"]
""" supported file extensions (lower case with dot). """


def check_image(fname, poller):
    """
    Check method that ensures the image is valid.

    :param fname: the file to check
    :type fname: str
    :param poller: the Poller instance that called the method
    :type poller: Poller
    :return: True if complete
    :rtype: bool
    """
    result = auto.is_image_complete(fname)
    poller.debug("Image complete:", fname, "->", result)
    return result


def process_image(fname, output_dir, poller):
    """
    Method for processing an image.

    :param fname: the image to process
    :type fname: str
    :param output_dir: the directory to write the image to
    :type output_dir: str
    :param poller: the Poller instance that called the method
    :type poller: Poller
    :return: the list of generated output files
    :rtype: list
    """
    result = []
    model_params = poller.params.model_params
    try:
        json_path = "{}/{}{}".format(output_dir, os.path.splitext(os.path.basename(fname))[0], ".json")
        img = Image.open(fname).resize((model_params["width"], model_params["height"]))
        preds = predict_image(img, model_params)
        with open(json_path, "w") as fp:
            json.dump(preds, fp, indent=2)
        result.append(json_path)
    except KeyboardInterrupt:
        poller.keyboard_interrupt()
    except:
        poller.error("Failed to process image: %s\n%s" % (fname, traceback.format_exc()))
    return result


def predict_on_images(model, labels, input_dir, output_dir, tmp_dir=None,
                      poll_wait=1.0, continuous=False, use_watchdog=False, watchdog_check_interval=10.0,
                      delete_input=False, input_mean=127.5, input_std=127.5, num_threads=None, top_x=5,
                      verbose=False, quiet=False):
    """
    Performs predictions on images found in input_dir and outputs the prediction PNG files in output_dir.

    :param model: the tflite model file to use
    :param model: str
    :param labels: the list of class labels
    :type labels: list
    :param input_dir: the directory with the images
    :type input_dir: str
    :param output_dir: the output directory to move the images to and store the predictions
    :type output_dir: str
    :param tmp_dir: the temporary directory to store the predictions until finished
    :type tmp_dir: str
    :param poll_wait: the amount of seconds between polls when not in watchdog mode
    :type poll_wait: float
    :param continuous: whether to poll for files continuously
    :type continuous: bool
    :param use_watchdog: whether to react to file creation events rather than use fixed-interval polling
    :type use_watchdog: bool
    :param watchdog_check_interval: the interval for the watchdog process to check for files that were missed due to potential race conditions
    :type watchdog_check_interval: float
    :param delete_input: whether to delete the input images rather than moving them to the output directory
    :type delete_input: bool
    :param input_mean: the input mean to use
    :type input_mean: float
    :param input_std: the input standard deviation to use
    :type input_std: float
    :param num_threads: the number of threads to use
    :type num_threads: int
    :param top_x: the number of labels with the highest probabilities to return, <1 for all
    :type top_x: int
    :param verbose: whether to output more logging information
    :type verbose: bool
    :param quiet: whether to suppress output
    :type quiet: bool
    """

    if verbose:
        print("Loading model: %s" % model)
    model_params = load_model(model, num_threads=num_threads)
    model_params["input_mean"] = input_mean
    model_params["input_std"] = input_std
    model_params["top_x"] = top_x
    model_params["label"] = labels

    poller = Poller()
    poller.input_dir = input_dir
    poller.output_dir = output_dir
    poller.tmp_dir = tmp_dir
    poller.extensions = SUPPORTED_EXTS
    poller.delete_input = delete_input
    poller.verbose = verbose
    poller.progress = not quiet
    poller.check_file = check_image
    poller.process_file = process_image
    poller.poll_wait = poll_wait
    poller.continuous = continuous
    poller.use_watchdog = use_watchdog
    poller.watchdog_check_interval = watchdog_check_interval
    poller.params.model_params = model_params
    poller.poll()


def main(args=None):
    """
    The main method for parsing command-line arguments and starting the training.

    :param args: the commandline arguments, uses sys.argv if not supplied
    :type args: list
    """

    parser = argparse.ArgumentParser(
        description="tflite Image Classification Prediction (file-polling)",
        prog="tf_predict_poll",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', metavar="FILE", type=str, required=True, help='The tflite model to use.')
    parser.add_argument('--labels', metavar="FILE", type=str, required=True, help='The text file with the labels (one per line).')
    parser.add_argument('--prediction_in', help='Path to the test images', required=True, default=None)
    parser.add_argument('--prediction_out', help='Path to the output csv files folder', required=True, default=None)
    parser.add_argument('--prediction_tmp', help='Path to the temporary csv files folder', required=False, default=None)
    parser.add_argument('--prediction_suffix', metavar='SUFFIX', help='The suffix to use for the prediction files', default="-rois.csv", required=False)
    parser.add_argument('--poll_wait', type=float, help='poll interval in seconds when not using watchdog mode', required=False, default=1.0)
    parser.add_argument('--continuous', action='store_true', help='Whether to continuously load test images and perform prediction', required=False, default=False)
    parser.add_argument('--use_watchdog', action='store_true', help='Whether to react to file creation events rather than performing fixed-interval polling', required=False, default=False)
    parser.add_argument('--watchdog_check_interval', type=float, help='check interval in seconds for the watchdog', required=False, default=10.0)
    parser.add_argument('--delete_input', action='store_true', help='Whether to delete the input images rather than move them to --prediction_out directory', required=False, default=False)
    parser.add_argument('--input_mean', metavar="MEAN", type=float, required=False, default=127.5, help='The input mean to use.')
    parser.add_argument('--input_std', metavar="STD", type=float, required=False, default=127.5, help='The input standard deviation to use.')
    parser.add_argument('--num_threads', metavar="INT", type=int, required=False, default=None, help='The number of threads to use.')
    parser.add_argument("--top_x", metavar="INT", type=int, help="output only the top K labels; use <1 for all", default=5)
    parser.add_argument('--verbose', action='store_true', help='Whether to output more logging info', required=False, default=False)
    parser.add_argument('--quiet', action='store_true', help='Whether to suppress output', required=False, default=False)
    parsed = parser.parse_args(args=args)

    print("Loading labels...")
    labels = load_labels(parsed.labels)

    predict_on_images(parsed.model, labels, parsed.prediction_in, parsed.prediction_out, tmp_dir=parsed.prediction_tmp,
                      poll_wait=parsed.poll_wait, continuous=parsed.continuous,
                      use_watchdog=parsed.use_watchdog, watchdog_check_interval=parsed.watchdog_check_interval,
                      delete_input=parsed.delete_input, input_mean=parsed.input_mean, input_std=parsed.input_std,
                      num_threads=parsed.iou_threshold, top_x=parsed.top_x, verbose=parsed.verbose, quiet=parsed.quiet)


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
