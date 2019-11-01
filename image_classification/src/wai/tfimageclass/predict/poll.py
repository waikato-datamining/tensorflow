# Copyright 2019 University of Waikato, Hamilton, NZ.
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
from wai.tfimageclass.utils.prediction_utils import load_graph, load_labels, read_tensor_from_image_file, tensor_to_probs, top_k_probs


def poll(sess, graph, input_layer, output_layer, labels, in_dir, out_dir, height, width, mean, std, top_x, delete):
    """
    Performs continuous predictions on files appearing in the "in_dir" and outputting the results in "out_dir".

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
    :param in_dir: the input directory to poll
    :type in_dir: str
    :param out_dir: the output directory for the results
    :type out_dir: str
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
                probs = tensor_to_probs(graph, input_layer, output_layer, tensor, sess)
                top_probs = top_k_probs(probs, top_x)
                with open(roi_tmp, "w") as rf:
                    rf.write("label,probability\n")
                    for i in top_probs:
                        rf.write(labels[i] + "," + str(probs[top_probs[i]]) + "\n")
                os.rename(roi_tmp, roi_csv)
            except Exception as e:
                print(traceback.format_exc())

            timediff = datetime.now() - start
            print("  time:", timediff)

        # nothing processed at all, lets wait for files to appear
        if not any:
            sleep(1)


def main(args=None):
    """
    The main method for parsing command-line arguments and labeling.

    :param args: the commandline arguments, uses sys.argv if not supplied
    :type args: list
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", help="the input directory to poll for images", required=True)
    parser.add_argument("--out_dir", help="the output directory for processed images and predictions", required=True)
    parser.add_argument('--delete', default=False, help="Whether to delete images rather than move them to the output directory.", action='store_true')
    parser.add_argument("--graph", help="graph/model to be executed", required=True)
    parser.add_argument("--labels", help="name of file containing labels", required=True)
    parser.add_argument("--input_height", type=int, help="input height", default=299)
    parser.add_argument("--input_width", type=int, help="input width", default=299)
    parser.add_argument("--input_mean", type=int, help="input mean", default=0)
    parser.add_argument("--input_std", type=int, help="input std", default=255)
    parser.add_argument("--input_layer", help="name of input layer", default="Placeholder")
    parser.add_argument("--output_layer", help="name of output layer", default="final_result")
    parser.add_argument("--top_x", type=int, help="output only the top K labels; use <1 for all", default=5)
    args = parser.parse_args(args=args)

    graph = load_graph(args.graph)
    labels = load_labels(args.labels)

    with tf.compat.v1.Session(graph=graph) as sess:
        poll(sess, graph, args.input_layer, args.output_layer, labels, args.in_dir, args.out_dir,
             args.input_height, args.input_width, args.input_mean, args.input_std, args.top_x, args.delete)


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
