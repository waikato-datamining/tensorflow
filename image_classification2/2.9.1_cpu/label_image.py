# Copyright (C) Tensorflow team
# Copyright (C) University of Waikato, Hamilton, NZ
# License Apache 2.0
# https://github.com/tensorflow/tensorflow/blob/950ae6aba94a46da7565d903f8b1328e32faa8fc/tensorflow/lite/examples/python/label_image.py
import argparse
import time

import numpy as np
import tensorflow as tf
from predict_utils import load_image, load_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--image',
        default='/tmp/grace_hopper.bmp',
        help='image to be classified')
    parser.add_argument(
        '-m',
        '--model_file',
        default='/tmp/mobilenet_v1_1.0_224_quant.tflite',
        help='.tflite model to be executed')
    parser.add_argument(
        '-l',
        '--label_file',
        default='/tmp/labels.txt',
        help='name of file containing labels')
    parser.add_argument(
        '--input_mean',
        default=127.5, type=float,
        help='input_mean')
    parser.add_argument(
        '--input_std',
        default=127.5, type=float,
        help='input standard deviation')
    parser.add_argument(
        '--num_threads', default=None, type=int, help='number of threads')
    parser.add_argument(
        '-e', '--ext_delegate', help='external_delegate_library path')
    parser.add_argument(
        '-o',
        '--ext_delegate_options',
        help='external delegate options, \
            format: "option1: value1; option2: value2"')

    args = parser.parse_args()

    ext_delegate = None
    ext_delegate_options = {}

    # parse extenal delegate options
    if args.ext_delegate_options is not None:
        options = args.ext_delegate_options.split(';')
        for o in options:
            kv = o.split(':')
            if (len(kv) == 2):
                ext_delegate_options[kv[0].strip()] = kv[1].strip()
            else:
                raise RuntimeError('Error parsing delegate option: ' + o)

    # load external delegate
    if args.ext_delegate is not None:
        print('Loading external delegate from {} with args: {}'.format(
            args.ext_delegate, ext_delegate_options))
        ext_delegate = [
            tflite.load_delegate(args.ext_delegate, ext_delegate_options)
        ]

    interpreter = tf.lite.Interpreter(
        model_path=args.model_file,
        experimental_delegates=ext_delegate,
        num_threads=args.num_threads)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    img = load_image(args.image, width, height)

    # add N dim
    input_data = np.expand_dims(img, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - args.input_mean) / args.input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)

    start_time = time.time()
    interpreter.invoke()
    stop_time = time.time()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(args.label_file)
    for i in top_k:
        if floating_model:
            print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
        else:
            print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))

    print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))

