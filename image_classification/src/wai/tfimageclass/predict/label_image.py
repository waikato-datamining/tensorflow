# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
import traceback
import tensorflow as tf
from collections import OrderedDict

from wai.tfimageclass.utils.prediction_utils import load_graph, load_tflite, load_labels, \
    read_tensor_from_image_file, read_tflite_tensor_from_image_file, tensor_to_probs, tflite_tensor_to_probs, \
    top_k_probs, tflite_top_k_probs, load_info_file, output_predictions


def main(args=None):
    """
    The main method for parsing command-line arguments and labeling.

    :param args: the commandline arguments, uses sys.argv if not supplied
    :type args: list
    """
    parser = argparse.ArgumentParser(
        description="Outputs predictions for single image using a trained model.",
        prog="tfic-labelimage",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image", help="image to be processed", required=True)
    parser.add_argument("--graph", help="graph/model to be executed", required=True)
    parser.add_argument("--graph_type", metavar="TYPE", choices=["tensorflow", "tflite"], help="the type of graph/model to be loaded", default="tensorflow", required=False)
    parser.add_argument("--info", help="name of json file with model info (dimensions, layers); overrides input_height/input_width/labels/input_layer/output_layer options", default=None)
    parser.add_argument("--labels", help="name of file containing labels", required=False)
    parser.add_argument("--input_height", type=int, help="input height", default=299)
    parser.add_argument("--input_width", type=int, help="input width", default=299)
    parser.add_argument("--input_layer", help="name of input layer", default="Placeholder")
    parser.add_argument("--output_layer", help="name of output layer", default="final_result")
    parser.add_argument("--input_mean", type=int, help="input mean", default=0)
    parser.add_argument("--input_std", type=int, help="input std", default=255)
    parser.add_argument("--top_x", type=int, help="output only the top K labels; use <1 for all", default=5)
    parser.add_argument("--output_format", metavar="TYPE", choices=["plaintext", "txt", "csv", "xml", "json"], help="the output format for the predictions", default="plaintext", required=False)
    parser.add_argument("--output_file", metavar="FILE", help="the file to write the predictions, uses stdout if not provided", default=None, required=False)
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

    if args.output_file is None:
        if args.top_x > 0:
            print("Top " + str(args.top_x) + " labels")
        else:
            print("All labels")

    predictions = OrderedDict()

    if args.graph_type == "tensorflow":
        graph = load_graph(args.graph)

        with tf.compat.v1.Session(graph=graph) as sess:
            tensor = read_tensor_from_image_file(
                args.image,
                input_height=input_height,
                input_width=input_width,
                input_mean=args.input_mean,
                input_std=args.input_std,
                sess=sess)
            results = tensor_to_probs(graph, input_layer, output_layer, tensor, sess)
            top_x = top_k_probs(results, args.top_x)
            for i in top_x:
                predictions[labels[i]] = results[i]
    elif args.graph_type == "tflite":
        interpreter = load_tflite(args.graph)
        tensor = read_tflite_tensor_from_image_file(args.image, input_height, input_width,
                                                    input_mean=args.input_mean, input_std=args.input_std)
        results = tflite_tensor_to_probs(interpreter, tensor)
        top_x = tflite_top_k_probs(results, args.top_x)
        for i in range(len(top_x)):
            predictions[labels[top_x[i]]] = results[0][top_x[i]]
    else:
        raise Exception("Unhandled graph type: %s" % args.graph_type)

    info = dict()
    info["model"] = args.graph
    output_predictions(predictions, output_file=args.output_file, output_format=args.output_format, info=info)


def sys_main() -> int:
    """
    Runs the main function using the system cli arguments, and
    returns a system error code.

    :return: 0 for success, 1 for failure.
    """
    try:
        main()
        return 0
    except Exception:
        print(traceback.format_exc())
        return 1


if __name__ == '__main__':
    main()
