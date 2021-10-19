import argparse
import json
import traceback

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier

from wai.tmm.common.hyper import load_hyper_parameters
from wai.tmm.common.io import model_path_name
from wai.tmm.common.optimize import OPTIMIZATIONS, OPTIMIZATION_NONE, configure_optimization


def write_labels(data, output_dir):
    """
    Writes the labels to disk.

    :param data: the training data to get the labels from
    :param output_dir: the output directory to store the labels in (labels.txt)
    """
    with open(output_dir + "/labels.txt", "w") as f:
        for l in data.index_to_label:
            f.write(l)
            f.write("\n")


def train(model_type, image_dir, output, num_epochs=None, hyper_params=None, batch_size=8,
          validation=0.15, testing=0.15, optimization=OPTIMIZATION_NONE, results=None):
    """
    Trains an object detection model.

    :param model_type: the model type, e.g., efficientdet_lite0
    :type model_type: str
    :param image_dir: the directory with images to use for training, validating, testing (sub-dirs act as classes)
    :type image_dir: str
    :param output: the directory or filename to store the model under (uses model.tflite if dir)
    :type output: str
    :param num_epochs: the number of epochs to use (default is 50), overrides num_epochs in hyper_params
    :type num_epochs: int
    :param hyper_params: the hyper parameters to override model's default ones with
    :type hyper_params: dict
    :param batch_size: the batch size to use for training
    :type batch_size: int
    :param validation: the percentage to use for validation (0-1)
    :type validation: float
    :param testing: the percentage to use for testing (0-1)
    :type testing: float
    :param optimization: how to optimize the model when saving it
    :type optimization: str
    :param results: the JSON file to store the evaluation results in
    :type results: str
    """
    hyper_params = load_hyper_parameters(hyper_params)
    if num_epochs is not None:
        hyper_params["epochs"] = num_epochs

    data = image_classifier.DataLoader.from_folder(image_dir, shuffle=True)
    train_data, val_test_data = data.split(1.0 - (validation + testing))
    if testing > 0:
        validation_data, test_data = val_test_data.split(validation / (validation + testing))
    else:
        validation_data = val_test_data
        test_data = None
    model = image_classifier.create(train_data, model_spec=model_spec.get(model_type), batch_size=batch_size,
                                    validation_data=validation_data, **hyper_params)
    output_dir, output_name = model_path_name(output)
    model.export(export_dir=output_dir, tflite_filename=output_name, quantization_config=configure_optimization(optimization))
    write_labels(train_data, output_dir)
    if test_data is not None:
        res = model.evaluate(test_data)
        print("Results on test data:")
        print("- loss: %.3f" % res[0])
        print("- accuracy: %.3f" % res[1])
        if results is not None:
            d = {}
            d["loss"] = float(res[0])
            d["accuracy"] = float(res[1])
            with open(results, "w") as f:
                json.dump(d, f, indent=2)


def main(args=None):
    """
    The main method for parsing command-line arguments and starting the training.

    :param args: the commandline arguments, uses sys.argv if not supplied
    :type args: list
    """

    parser = argparse.ArgumentParser(
        description="Trains a tflite image classification model.\n"
                    + "For hyper parameters, see:\n"
                    + "https://www.tensorflow.org/lite/tutorials/model_maker_image_classification",
        prog="tmm-ic-train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--images', metavar="DIR", type=str, required=True, help='The directory with images (with sub-dirs containing images for separate class).')
    parser.add_argument('--model_type', type=str, choices=model_spec.IMAGE_CLASSIFICATION_MODELS, default="efficientnet_lite0", help='The model architecture to use.')
    parser.add_argument('--hyper_params', metavar="FILE", type=str, required=False, help='The YAML file with hyper parameter settings.')
    parser.add_argument('--num_epochs', metavar="INT", type=int, default=None, help='The number of epochs to use for training (can also be supplied through hyper parameters).')
    parser.add_argument('--batch_size', metavar="INT", type=int, default=8, help='The batch size to use.')
    parser.add_argument('--output', metavar="DIR_OR_FILE", type=str, required=True, help='The directory or filename to store the model under (uses model.tflite if dir). The labels gets stored in "labels.txt" in the determined directory.')
    parser.add_argument('--optimization', type=str, choices=OPTIMIZATIONS, default=OPTIMIZATION_NONE, help='How to optimize the model when saving it.')
    parser.add_argument('--validation', metavar="0-1", type=float, default=0.15, help='The dataset percentage to use for validation.')
    parser.add_argument('--testing', metavar="0-1", type=float, default=0.15, help='The dataset percentage to use for testing.')
    parser.add_argument('--results', metavar="FILE", type=str, default=None, help='The JSON file to store the evaluation results in (requires --testing).')
    parsed = parser.parse_args(args=args)

    train(parsed.model_type, parsed.images, parsed.output, num_epochs=parsed.num_epochs,
          hyper_params=parsed.hyper_params, batch_size=parsed.batch_size,
          validation=parsed.validation, testing=parsed.testing, optimization=parsed.optimization,
          results=parsed.results)


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
