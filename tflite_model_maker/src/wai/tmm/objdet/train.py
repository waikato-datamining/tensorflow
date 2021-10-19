import argparse
import json
import traceback

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

from wai.tmm.common.hyper import add_hyper_parameters
from wai.tmm.common.io import model_path_name
from wai.tmm.common.optimize import OPTIMIZATIONS, OPTIMIZATION_NONE, configure_optimization


def write_labels(data, output_dir):
    """
    Writes the labels to disk.
    
    :param data: the training data to get the labels from
    :param output_dir: the output directory to store the labels in (labels.json, labels.txt)
    """
    keys = list(data.label_map.keys())
    keys.sort()
    labels = {}
    for k in keys:
        labels[k] = str(data.label_map[k])
    with open(output_dir + "/labels.json", "w") as f:
        json.dump(labels, f)
    with open(output_dir + "/labels.txt", "w") as f:
        for k in keys:
            f.write(labels[k])
            f.write("\n")


def train(model_type, annotations, output, num_epochs=None, hyper_params=None, batch_size=8, evaluate=False,
          optimization=OPTIMIZATION_NONE, results=None):
    """
    Trains an object detection model.

    :param model_type: the model type, e.g., efficientdet_lite0
    :type model_type: str
    :param annotations: the CSV file with annotations to use for trainin/validation
    :type annotations: str
    :param output: the directory or filename to store the model under (uses model.tflite if dir)
    :type output: str
    :param num_epochs: the number of epochs to use (default is 50), overrides num_epochs in hyper_params
    :type num_epochs: int
    :param hyper_params: the hyper parameters to override model's default ones with
    :type hyper_params: dict
    :param batch_size: the batch size to use for training
    :type batch_size: int
    :param evaluate: whether to evaluate the model if there is a test dataset in the data
    :type evaluate: bool
    :param optimization: how to optimize the model when saving it
    :type optimization: str
    :param results: the JSON file to store the evaluation results in
    :type results: str
    """
    spec = model_spec.get(model_type)
    add_hyper_parameters(spec, hyper_params)
    if num_epochs is not None:
        spec.config.num_epochs = num_epochs

    output_dir, output_name = model_path_name(output)
    train_data, validation_data, test_data = object_detector.DataLoader.from_csv(annotations)
    model = object_detector.create(train_data, model_spec=spec, batch_size=batch_size, train_whole_model=True,
                                   validation_data=validation_data)
    model.export(export_dir=output_dir, tflite_filename=output_name, quantization_config=configure_optimization(optimization))
    write_labels(train_data, output_dir)
    if evaluate:
        res = model.evaluate(test_data)
        d = {}
        for k in res:
            print("%s: %.2f" % (k, res[k]))
            d[str(k)] = float(res[k])
        if results is not None:
            with open(results, "w") as f:
                json.dump(d, f, indent=2)


def main(args=None):
    """
    The main method for parsing command-line arguments and starting the training.

    :param args: the commandline arguments, uses sys.argv if not supplied
    :type args: list
    """

    parser = argparse.ArgumentParser(
        description="Trains a tflite object detection model.\n"
                    + "For hyper parameters, see:\n"
                    + "https://www.tensorflow.org/lite/tutorials/model_maker_object_detection",
        prog="tmm-od-train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--annotations', metavar="FILE", type=str, required=True, help='The CSV file with the annotations.')
    parser.add_argument('--model_type', type=str, choices=model_spec.OBJECT_DETECTION_MODELS, default="efficientdet_lite0", help='The model architecture to use.')
    parser.add_argument('--hyper_params', metavar="FILE", type=str, required=False, help='The YAML file with hyper parameter settings.')
    parser.add_argument('--num_epochs', metavar="INT", type=int, default=None, help='The number of epochs to use for training (can also be supplied through hyper parameters).')
    parser.add_argument('--batch_size', metavar="INT", type=int, default=8, help='The batch size to use.')
    parser.add_argument('--output', metavar="DIR_OR_FILE", type=str, required=True, help='The directory or filename to store the model under (uses model.tflite if dir). The labels gets stored in "labels.txt" in the determined directory.')
    parser.add_argument('--optimization', type=str, choices=OPTIMIZATIONS, default=OPTIMIZATION_NONE, help='How to optimize the model when saving it.')
    parser.add_argument('--evaluate', action="store_true", help='If test data is part of the annotations, then the resulting model can be evaluated against it.')
    parser.add_argument('--results', metavar="FILE", type=str, default=None, help='The JSON file to store the evaluation results in.')
    parsed = parser.parse_args(args=args)

    train(parsed.model_type, parsed.annotations, parsed.output, num_epochs=parsed.num_epochs,
          hyper_params=parsed.hyper_params, batch_size=parsed.batch_size, evaluate=parsed.evaluate,
          optimization=parsed.optimization, results=parsed.results)


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
