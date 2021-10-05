import os
import tensorflow as tf


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


def model_path_name(output):
    """
    Generates output dir and output name from the input, which can be either a directory (using model.tflite then)
    or a filename.

    :param output: the dir/file to use to determine output dir and name
    :type output: str
    :return: the tuple of path, name
    :rtype: tuple
    """
    if os.path.isdir(output):
        output_dir = output
        output_name = "model.tflite"
    else:
        output_dir = os.path.dirname(output)
        output_name = os.path.basename(output)
    return output_dir, output_name
