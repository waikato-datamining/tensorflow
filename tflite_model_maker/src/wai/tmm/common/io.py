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