import argparse
import traceback
import yaml

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

from tflite_model_maker import model_spec
from tflite_model_maker import object_detector


def train(model_type, annotations, output_dir, num_epochs=None, hyper_params=None, batch_size=8, evaluate=False):
    """
    Trains an object detection model.

    :param model_type: the model type, e.g., efficientdet_lite0
    :type model_type: str
    :param annotations: the CSV file with annotations to use for trainin/validation
    :type annotations: str
    :param output_dir: the output directory to store the model in
    :type output_dir: str
    :param num_epochs: the number of epochs to use (default is 50), overrides num_epochs in hyper_params
    :type num_epochs: int
    :param hyper_params: the hyper parameters to override model's default ones with
    :type hyper_params: dict
    :param batch_size: the batch size to use for training
    :type batch_size: int
    :param evaluate: whether to evaluate the model if there is a test dataset in the data
    :type evaluate: bool
    """
    spec = model_spec.get(model_type)
    if hyper_params is not None:
        for k in hyper_params:
            setattr(spec.config, k, hyper_params[k])
    if num_epochs is not None:
        spec.config.num_epochs = num_epochs

    train_data, validation_data, test_data = object_detector.DataLoader.from_csv(annotations)
    model = object_detector.create(train_data, model_spec=spec, batch_size=batch_size, train_whole_model=True,
                                   validation_data=validation_data)
    model.export(export_dir=output_dir)
    if evaluate:
        results = model.evaluate(test_data)
        for k in results:
            print("%s: %.2f" % (k, results[k]))


def main(args=None):
    """
    The main method for parsing command-line arguments and starting the training.

    :param args: the commandline arguments, uses sys.argv if not supplied
    :type args: list
    """

    parser = argparse.ArgumentParser(
        description="Trains an object detection model.",
        prog="tmm-od-train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--annotations', metavar="FILE", type=str, required=True, help='The CSV file with the annotations.')
    parser.add_argument('--model_type', type=str, choices=model_spec.OBJECT_DETECTION_MODELS, default="efficientdet_lite0", help='The model architecture to use.')
    parser.add_argument('--hyper_params', metavar="FILE", type=str, required=False, help='The YAML file with hyper parameter settings.')
    parser.add_argument('--num_epochs', metavar="INT", type=int, default=None, help='The number of epochs to use for training (can also be supplied through hyper parameters).')
    parser.add_argument('--batch_size', metavar="INT", type=int, default=8, help='The batch size to use.')
    parser.add_argument('--output_dir', metavar="DIR", type=str, required=True, help='The directory to store the trained model in.')
    parser.add_argument('--evaluate', action="store_true", help='If test data is part of the annotations, then the resulting model can be evaluated against it.')
    parsed = parser.parse_args(args=args)

    hyper_params = None
    if parsed.hyper_params is not None:
        with open(parsed.hyper_params, "r") as f:
            hyper_params = yaml.safe_load(f)

    train(parsed.model_type, parsed.annotations, parsed.output_dir, num_epochs=parsed.num_epochs,
          hyper_params=hyper_params, batch_size=parsed.batch_size, evaluate=parsed.evaluate)


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
