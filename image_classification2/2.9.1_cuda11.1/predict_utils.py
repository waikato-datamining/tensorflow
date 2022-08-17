import io
import numpy as np
import tensorflow as tf
from PIL import Image


def load_labels(labels_file):
    """
    Loads the labels from the specified file.

    :param labels_file: the file to load (one label per line)
    :type labels_file: str
    :return: the list of labels
    :rtype: list
    """
    with open(labels_file, "r") as f:
        return [line.strip() for line in f.readlines()]


def load_model(model_file, num_threads=None):
    """
    Loads the model from disk.

    :param model_file: the tflite model to load
    :type model_file: str
    :param num_threads: the number of threads to use
    :type num_threads: int
    :return: the dictionary with the model and parameters determined
    :rtype: dict
    """
    interpreter = tf.lite.Interpreter(model_path=model_file, num_threads=num_threads)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    floating_model = input_details[0]['dtype'] == np.float32
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    return {
        "interpreter": interpreter,
        "input_details": input_details,
        "output_details": output_details,
        "floating_model": floating_model,
        "width": width,
        "height": height,
    }


def remove_alpha_channel(img):
    """
    Removes any alpha channel from the image.

    :param img: the image to process
    :return: the image without the alpha channel
    """
    if img.mode == "RGBA":
        img = img.convert("RGB")
    return img


def load_image(fname, width, height):
    """
    Loads the image from the specified file.

    :param fname: the name of the file to load
    :type fname: str
    :param width: the width of the image
    :type width: int
    :param height: the height of the image
    :type height: int
    :return: the image
    """
    img = Image.open(fname).resize((width, height))
    return remove_alpha_channel(img)


def read_image(data, width, height):
    """
    Reads the image from the bytes.

    :param data: the bytes to read the image from
    :type data: bytes
    :param width: the width of the image
    :type width: int
    :param height: the height of the image
    :type height: int
    :return: the image
    """
    img = Image.open(io.BytesIO(data)).resize((width, height))
    return remove_alpha_channel(img)


def predict_image(img, model_params):
    """
    Performs a prediction on the image object.

    :param img: the pillow image to predict on
    :type img: PIL.Image
    :param model_params: the model parameters to use
    :type model_params: dict
    :return: the dictionary with label/probability relation
    :rtype: dict
    """

    # prepare input data
    # add N dim
    input_data = np.expand_dims(img, axis=0)
    if model_params["floating_model"]:
        input_data = (np.float32(input_data) - model_params["input_mean"]) / model_params["input_std"]

    # make prediction
    model_params["interpreter"].set_tensor(model_params["input_details"][0]['index'], input_data)
    model_params["interpreter"].invoke()
    output_data = model_params["interpreter"].get_tensor(model_params["output_details"][0]['index'])
    preds = np.squeeze(output_data)

    # top X
    if model_params["top_x"] < 1:
        top_k = preds.argsort()[:][::-1]
    else:
        top_k = preds.argsort()[-model_params["top_x"]:][::-1]

    # assemble output
    result = {}
    for i in top_k:
        if model_params["floating_model"]:
            result[model_params["labels"][i]] = float(preds[i])
        else:
            result[model_params["labels"][i]] = float(preds[i] / 255.0)

    return result
