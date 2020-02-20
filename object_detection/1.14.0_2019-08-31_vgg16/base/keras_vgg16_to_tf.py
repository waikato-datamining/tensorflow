from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np
from keras.models import model_from_json # or from tf.keras.models import model_from_json
import tensorflow as tf
import argparse


def VGG_16():
    """
    Generates the model structure.

    Taken from here:
    https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3#file-vgg-16_keras-py
    """

    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    return model


def keras_ckpt_to_tf_ckpt(keras_weights_path,
                          tf_ckpt_save_path,
                          keras_model_json_path=None):
    """ 
    Function to convert a keras classification checkpoint to tensorflow checkpoint
    To use in tensorflow object detection API

    Taken from here:
    https://abhijit-2592.github.io/Keras-with-TFODAPI/
    
    :param keras_weights_path: full path to .h5 or hdf5 file
    :type keras_weights_path: str
    :param tf_ckpt_save_path: full path to save the converted tf checkpoint
    :type tf_ckpt_save_path: str
    :param keras_model_json_path: full path to .json file, calls VGG_16 if None
    :type keras_model_json_path: str
    """

    with tf.Session() as sess:
        # create/load model
        if keras_model_json_path is None:
            model = VGG_16()
        else:
            with open(keras_model_json_path,'r') as json_file:
                loaded_model_json = json_file.read()
                json_file.close()
                model = model_from_json(loaded_model_json)

        # load weights
        model.load_weights(keras_weights_path)
        print('loaded keras model from disk')
        saver = tf.train.Saver()
        saver.save(sess,tf_ckpt_save_path)
        print("Tensorflow checkpoint is saved in {}".format(tf_ckpt_save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--keras_weights', help='Path to the keras .h5 weights file', required=True, default=None)
    parser.add_argument('--tf_ckpt', help='Path to the tensorflow checkpoint file to save the converted model to', required=True, default=None)
    parser.add_argument('--keras_json', help='Path to the keras model description in JSON format', required=False, default=None)
    parsed = parser.parse_args()

    try:
        keras_ckpt_to_tf_ckpt(parsed.keras_weights, parsed.tf_ckpt, parsed.keras_json)
    except Exception as e:
        print(traceback.format_exc())

