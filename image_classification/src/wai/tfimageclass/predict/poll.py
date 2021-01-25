# Copyright 2019-2021 University of Waikato, Hamilton, NZ.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
from time import sleep
import os
import tensorflow as tf
import traceback
from wai.tfimageclass.utils.prediction_utils import load_graph, load_tflite, load_labels, \
    read_tensor_from_image_file, read_tflite_tensor_from_image_file, tensor_to_probs, tflite_tensor_to_probs, \
    top_k_probs, tflite_top_k_probs, load_info_file
import numpy as np


def predict_image(sess, graph, input_layer, output_layer, labels, top_x, tensor, output_file):
    """
    Obtains predictions for the image (in tensor representation) from the graph and outputs
    them in the output file.

    :param sess: the tensorflow session to use
    :type sess: tf.Session
    :param graph: the tensorflow graph to use
    :type graph: tf.Graph
    :param input_layer: the name of input layer in the graph to use
    :type input_layer: str
    :param output_layer: the name of output layer in the graph to use
    :type output_layer: str
    :param labels: the list of labels to use
    :type labels: list
    :param top_x: the number of labels with the highest probabilities to return, <1 for all
    :type top_x: int
    :param tensor: the image as tensor
    :type tensor: tf.Tensor
    :param output_file: the file to store the predictions in
    :type: str
    """

    probs = tensor_to_probs(graph, input_layer, output_layer, tensor, sess)
    top_probs = top_k_probs(probs, top_x)
    with open(output_file, "w") as rf:
        rf.write("label,probability\n")
        for i in top_probs:
            rf.write(labels[top_probs[i]] + "," + str(probs[top_probs[i]]) + "\n")


def poll(graph, input_layer, output_layer, labels, in_dir, out_dir, continuous, height, width, mean, std, top_x, delete,
         reset_session=50):
    """
    Performs continuous predictions on files appearing in the "in_dir" and outputting the results in "out_dir".

    :param graph: the tensorflow graph to use
    :type graph: tf.Graph
    :param input_layer: the name of input layer in the graph to use
    :type input_layer: str
    :param output_layer: the name of output layer in the graph to use
    :type output_layer: str
    :param labels: the list of labels to use
    :type labels: list
    :param in_dir: the input directory to poll
    :type in_dir: str
    :param out_dir: the output directory for the results
    :type out_dir: str
    :param continuous: whether to continuously poll for images or exit ones no more images to process
    :type continuous: bool
    :param height: the expected height of the images
    :type height: int
    :param width: the expected height of the images
    :type width: int
    :param mean: the mean to use for the images
    :type mean: int
    :param std: the std deviation to use for the images
    :type std: int
    :param top_x: the number of labels with the highest probabilities to return, <1 for all
    :type top_x: int
    :param delete: whether to delete the input images (True) or move them to the output directory (False)
    :type delete: bool
    :param reset_session: the number of processed images after which to reinitialize the session to avoid memory leaks
    :type reset_session: int
    """

    print("Class labels: %s" % str(labels))

    while True:
        num_processed = 0
        do_reset_session = False
        with tf.compat.v1.Session(graph=graph) as sess:
            while True:
                any = False
                files = [(in_dir + os.sep + x) for x in os.listdir(in_dir) if (x.lower().endswith(".png") or x.lower().endswith(".jpg"))]
                for f in files:
                    any = True
                    start = datetime.now()
                    print(start, "-", f)

                    img_path = out_dir + os.sep + os.path.basename(f)
                    roi_csv = out_dir + os.sep + os.path.splitext(os.path.basename(f))[0] + ".csv"
                    roi_tmp = out_dir + os.sep + os.path.splitext(os.path.basename(f))[0] + ".tmp"

                    tensor = None
                    try:
                        tensor = read_tensor_from_image_file(f, height, width, mean, std, sess)
                    except Exception as e:
                        print(traceback.format_exc())

                    try:
                        # delete any existing old files in output dir
                        if os.path.exists(img_path):
                            try:
                                os.remove(img_path)
                            except:
                                print("Failed to remove existing image in output directory: ", img_path)
                        if os.path.exists(roi_tmp):
                            try:
                                os.remove(roi_tmp)
                            except:
                                print("Failed to remove existing ROI file (tmp) in output directory: ", roi_tmp)
                        if os.path.exists(roi_csv):
                            try:
                                os.remove(roi_csv)
                            except:
                                print("Failed to remove existing ROI file in output directory: ", roi_csv)
                        # delete or move into output dir
                        if delete:
                            os.remove(f)
                        else:
                            os.rename(f, img_path)
                    except:
                        img_path = None

                    if tensor is None:
                        continue
                    if img_path is None:
                        continue

                    try:
                        predict_image(sess, graph, input_layer, output_layer, labels, top_x, tensor, roi_tmp)
                        os.rename(roi_tmp, roi_csv)
                        num_processed += 1
                    except Exception:
                        print(traceback.format_exc())

                    timediff = datetime.now() - start
                    print("  time:", timediff)

                    if num_processed >= reset_session:
                        print("\nResetting session...\n")
                        do_reset_session = True
                        break

                # exit if not in continuous mode
                if not continuous:
                    return

                if do_reset_session:
                    break

                # nothing processed at all, lets wait for files to appear
                if not any:
                    sleep(1)


def tflite_predict_image(interpreter, labels, top_x, tensor, output_file):
    """
    Obtains predictions for the image (in tensor representation) from the tflite interpreter
    and outputs them in the output file.

    :param interpreter: the tflite interpreter to use
    :type interpreter: tf.lite.Interpreter
    :param labels: the list of labels to use
    :type labels: list
    :param top_x: the number of labels with the highest probabilities to return, <1 for all
    :type top_x: int
    :param tensor: the image as tensor
    :type tensor: tf.Tensor
    :param output_file: the file to store the predictions in
    :type: str
    """

    probs = tflite_tensor_to_probs(interpreter, tensor)
    top_probs = tflite_top_k_probs(probs, top_x)
    with open(output_file, "w") as rf:
        rf.write("label,probability\n")
        for i in range(len(top_probs)):
            rf.write(labels[top_probs[i]] + "," + str(probs[0][top_probs[i]]) + "\n")


def tflite_poll(interpreter, labels, in_dir, out_dir, continuous, height, width, mean, std, top_x, delete):
    """
    Performs continuous predictions on files appearing in the "in_dir" and outputting the results in "out_dir".

    :param interpreter: the tflite interpreter to use
    :type interpreter: tf.lite.Interpreter
    :param labels: the list of labels to use
    :type labels: list
    :param in_dir: the input directory to poll
    :type in_dir: str
    :param out_dir: the output directory for the results
    :type out_dir: str
    :param continuous: whether to continuously poll for images or exit ones no more images to process
    :type continuous: bool
    :param height: the expected height of the images
    :type height: int
    :param width: the expected height of the images
    :type width: int
    :param mean: the mean to use for the images
    :type mean: int
    :param std: the std deviation to use for the images
    :type std: int
    :param top_x: the number of labels with the highest probabilities to return, <1 for all
    :type top_x: int
    :param delete: whether to delete the input images (True) or move them to the output directory (False)
    :type delete: bool
    """

    print("Class labels: %s" % str(labels))

    while True:
        any = False
        files = [(in_dir + os.sep + x) for x in os.listdir(in_dir) if (x.lower().endswith(".png") or x.lower().endswith(".jpg"))]
        for f in files:
            any = True
            start = datetime.now()
            print(start, "-", f)

            img_path = out_dir + os.sep + os.path.basename(f)
            roi_csv = out_dir + os.sep + os.path.splitext(os.path.basename(f))[0] + ".csv"
            roi_tmp = out_dir + os.sep + os.path.splitext(os.path.basename(f))[0] + ".tmp"

            input_data = None
            try:
                input_data = read_tflite_tensor_from_image_file(f, height, width, input_mean=mean, input_std=std)
            except Exception:
                print(traceback.format_exc())

            try:
                # delete any existing old files in output dir
                if os.path.exists(img_path):
                    try:
                        os.remove(img_path)
                    except:
                        print("Failed to remove existing image in output directory: ", img_path)
                if os.path.exists(roi_tmp):
                    try:
                        os.remove(roi_tmp)
                    except:
                        print("Failed to remove existing ROI file (tmp) in output directory: ", roi_tmp)
                if os.path.exists(roi_csv):
                    try:
                        os.remove(roi_csv)
                    except:
                        print("Failed to remove existing ROI file in output directory: ", roi_csv)
                # delete or move into output dir
                if delete:
                    os.remove(f)
                else:
                    os.rename(f, img_path)
            except:
                img_path = None

            if input_data is None:
                continue
            if img_path is None:
                continue

            try:
                tflite_predict_image(interpreter, labels, top_x, input_data, roi_tmp)
                os.rename(roi_tmp, roi_csv)
            except Exception:
                print(traceback.format_exc())

            timediff = datetime.now() - start
            print("  time:", timediff)

        # exit if not in continuous mode
        if not continuous:
            return

        # nothing processed at all, lets wait for files to appear
        if not any:
            sleep(1)


def main(args=None):
    """
    The main method for parsing command-line arguments and labeling.

    :param args: the commandline arguments, uses sys.argv if not supplied
    :type args: list
    """
    parser = argparse.ArgumentParser(
        description="For bulk or continuous prediction output using a trained model.",
        prog="tfic-poll",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--in_dir", metavar="DIR", help="the input directory to poll for images", required=True)
    parser.add_argument("--out_dir", metavar="DIR", help="the output directory for processed images and predictions", required=True)
    parser.add_argument('--continuous', action='store_true', help='Whether to continuously load test images and perform prediction', required=False, default=False)
    parser.add_argument('--delete', default=False, help="Whether to delete images rather than move them to the output directory.", action='store_true')
    parser.add_argument("--graph", metavar="FILE", help="graph/model to be executed", required=True)
    parser.add_argument("--graph_type", metavar="TYPE", choices=["tensorflow", "tflite"], help="the type of graph/model to be loaded", default="tensorflow", required=False)
    parser.add_argument("--info", help="name of json file with model info (dimensions, layers); overrides input_height/input_width/labels/input_layer/output_layer options", default=None)
    parser.add_argument("--labels", metavar="FILE", help="name of file containing labels", required=False)
    parser.add_argument("--input_height", metavar="INT", type=int, help="input height", default=299)
    parser.add_argument("--input_width", metavar="INT", type=int, help="input width", default=299)
    parser.add_argument("--input_layer", metavar="NAME", help="name of input layer", default="Placeholder")
    parser.add_argument("--output_layer", metavar="NAME", help="name of output layer", default="final_result")
    parser.add_argument("--input_mean", metavar="INT", type=int, help="input mean", default=0)
    parser.add_argument("--input_std", metavar="INT", type=int, help="input std", default=255)
    parser.add_argument("--top_x", metavar="INT", type=int, help="output only the top K labels; use <1 for all", default=5)
    parser.add_argument("--reset_session", metavar="INT", type=int, help="The number of processed images after which to reinitialize the Tensorflow session to reduce memory leaks.", default=50)
    args = parser.parse_args(args=args)

    # values from options
    labels = None
    input_height = args.input_height
    input_width = args.input_width
    input_layer = args.input_layer
    output_layer = args.output_layer

    # override from info file?
    if args.info is not None:
        input_height, input_width, input_layer, output_layer, labels = load_info_file(args.info)

    if (labels is None) and (args.labels is not None):
        labels = load_labels(args.labels)
    if labels is None:
        raise Exception("No labels determined, either supply --info or --labels!")

    if args.graph_type == "tensorflow":
        graph = load_graph(args.graph)
        poll(graph, input_layer, output_layer, labels, args.in_dir, args.out_dir, args.continuous,
             input_height, input_width, args.input_mean, args.input_std, args.top_x, args.delete,
             reset_session=args.reset_session)
    elif args.graph_type == "tflite":
        interpreter = load_tflite(args.graph)
        tflite_poll(interpreter, labels, args.in_dir, args.out_dir, args.continuous,
                    input_height, input_width, args.input_mean, args.input_std, args.top_x, args.delete)
    else:
        raise Exception("Unhandled graph type: %s" % args.graph_type)


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
