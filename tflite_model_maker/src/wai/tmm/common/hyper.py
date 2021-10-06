import yaml

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)


def load_hyper_parameters(hyper_params):
    """
    Loads the file with hyper parameters.

    :param hyper_params: the YAML file with hyper parameters to load; if None returns an empty dict; if dict returns this
    :return: the dictionary of hyper parameters
    :rtype: dict
    """
    if isinstance(hyper_params, str):
        with open(hyper_params, "r") as f:
            return yaml.safe_load(f)
    if isinstance(hyper_params, dict):
        return hyper_params
    return {}


def add_hyper_parameters(model_spec, hyper_params, verbose=False):
    """
    Adds the hyper parameters to the model spec.

    :param model_spec: the model specification
    :param hyper_params: the file (str) or dictionary of hyper parameters to set, can be None
    :param verbose: whether to print debugging output
    :type verbose: bool
    """
    if hyper_params is None:
        if verbose:
            tf.get_logger().log(logging.INFO, "No hyper parameters to set")
        return
    hyper_params = load_hyper_parameters(hyper_params)
    if isinstance(hyper_params, dict):
        for k in hyper_params:
            if verbose:
                tf.get_logger().log(logging.INFO, "Setting hyper parameter: %s -> %s" % (str(k), str(hyper_params[k])))
            setattr(model_spec.config, k, hyper_params[k])
