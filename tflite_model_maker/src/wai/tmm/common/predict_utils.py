import numpy as np
import tensorflow as tf


def set_input_tensor(interpreter, image):
    """
    Set the input tensor.

    :param interpreter: the model to feed in the image
    :param image: the preprocessed image to feed in
    """
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
    """
    Return the output tensor at the given index.

    :param interpreter: the model to get the tensor from
    :param index: the index of the tensor
    :type index: int
    :return: the tensor
    :rtype: nd.ndarray
    """
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        tensor = scale * (tensor - zero_point)
    return tensor


def preprocess_image(image_path, input_size):
    """
    Preprocess the input image to feed to the TFLite model.

    :param image_path: the image to load
    :type image_path: str
    :param input_size: the tuple (height, width) to resize to
    :type input_size: tuple
    :return: the preprocessed image
    :rtype: np.ndarray
    """
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    original_image = img
    resized_img = tf.image.resize(img, input_size)
    resized_img = resized_img[tf.newaxis, :]
    return resized_img, original_image


def preprocess_image_bytes(image_data, input_size):
    """
    Preprocess the input image to feed to the TFLite model.

    :param image_data: the image bytes to load
    :type image_data: bytes
    :param input_size: the tuple (height, width) to resize to
    :type input_size: tuple
    :return: the preprocessed image
    :rtype: np.ndarray
    """
    img = tf.io.decode_image(image_data, channels=3)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    original_image = img
    resized_img = tf.image.resize(img, input_size)
    resized_img = resized_img[tf.newaxis, :]
    return resized_img, original_image