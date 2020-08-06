# Copyright 2019-2020 University of Waikato, Hamilton, NZ.
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


import argparse
import os
import traceback
import tensorflow as tf
from wai.tfimageclass.utils.train_utils import load_image_list, locate_sub_dirs, locate_images
from wai.tfimageclass.utils.prediction_utils import load_graph, read_tensor_from_image_file, tensor_to_probs, load_labels, top_k_probs, load_info_file
from wai.tfimageclass.utils.logging_utils import logging_level_verbosity


def init_counts(labels):
    """
    Initializes a dictionary with zeroed counts, using the labels as keys.

    :param labels: the list of labels to use as keys
    :type labels: list
    :return: the initialized counts
    :rtype: dict
    """

    result = dict()
    result[''] = 0
    for label in labels:
        result[label] = 0
    return result


def generate_stats(sess, graph, input_layer, output_layer, labels, image_dir, image_file_list, height, width, mean, std,
                   output_preds, output_stats, logging_verbosity):
    """
    Evaluates the built model on images form the specified directory, which can be limited to file listed
    in the image file list.

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
    :param image_dir: the directory with the images (sub-directories correspond to labels)
    :type image_dir: str
    :param image_file_list: the image file list to use (the keys correspond to labels, and the values contain the images w/o path); uses all images if None
    :type image_file_list: dict
    :param height: the expected height of the images
    :type height: int
    :param width: the expected height of the images
    :type width: int
    :param mean: the mean to use for the images
    :type mean: int
    :param std: the std deviation to use for the images
    :type std: int
    :param output_preds: the file to store the predictions in
    :type output_preds: str
    :param output_stats: the file to store the statistics in
    :type output_stats: str
    :param logging_verbosity: the level ('DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL')
    :type logging_verbosity: str
    """

    logging_verbosity = logging_level_verbosity(logging_verbosity)
    tf.compat.v1.logging.set_verbosity(logging_verbosity)

    tf.compat.v1.logging.info("Class labels: %s" % str(labels))

    if not tf.io.gfile.exists(image_dir):
        tf.compat.v1.logging.error("Image directory '" + image_dir + "' not found.")
        return None
    sub_dirs = locate_sub_dirs(image_dir)

    # compile lists of files per label
    if image_file_list:
        tf.compat.v1.logging.info("Using image list: %s" % image_file_list)
        image_list = load_image_list(image_file_list)
    else:
        image_list = dict()
        for label_name in sub_dirs:
            image_list[label_name] = locate_images(sub_dirs[label_name], strip_path=True)

    total = init_counts(labels)
    correct = init_counts(labels)
    incorrect = init_counts(labels)
    with open(output_preds, "w") as pf:
        pf.write("image,actual,predicted,error,probability\n")
        for label_name in sub_dirs:
            if label_name not in image_list:
                continue
            tf.compat.v1.logging.info(label_name)
            sub_dir = sub_dirs[label_name]
            file_list = image_list[label_name]
            count = 0
            for file_name in file_list:
                total[''] += 1
                total[label_name] += 1
                full_name = os.path.join(sub_dir, file_name)
                tensor = read_tensor_from_image_file(full_name, height, width, mean, std, sess)
                probs = tensor_to_probs(graph, input_layer, output_layer, tensor, sess)
                for i in top_k_probs(probs, 1):
                    pf.write("%s,%s,%s,%s,%f\n" %(full_name, label_name, labels[i], label_name != labels[i], probs[i]))
                    if label_name != labels[i]:
                        incorrect[''] += 1
                        incorrect[label_name] += 1
                    else:
                        correct[''] += 1
                        correct[label_name] += 1
                # progress
                count += 1
                if count % 10 == 0:
                    tf.compat.v1.logging.info("%d / %d" % (count, len(file_list)))

    with open(output_stats, "w") as sf:
        sf.write("statistic,value\n")
        keys = sorted(total.keys())
        for key in keys:
            if key == '':
                prefix = "total - "
            else:
                prefix = key + " - "
            num_total = total[key]
            num_correct = correct[key]
            num_incorrect = incorrect[key]
            if num_total > 0:
                acc = num_correct / num_total
            else:
                acc = float("NaN")
            sf.write("%s%s,%d\n" % (prefix, "number of images", num_total))
            sf.write("%s%s,%d\n" % (prefix, "number of correct predictions", num_correct))
            sf.write("%s%s,%d\n" % (prefix, "number of incorrect predictions", num_incorrect))
            sf.write("%s%s,%f\n" % (prefix, "accuracy", acc))


def main(args=None):
    """
    The main method for parsing command-line arguments and labeling.

    :param args: the commandline arguments, uses sys.argv if not supplied
    :type args: list
    """
    parser = argparse.ArgumentParser(
        description="Generates statistics in CSV format by recording predictions on images list files.",
        prog="tfic-stats",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image_dir', type=str, default='', help='Path to folders of labeled images.')
    parser.add_argument('--image_list', type=str, required=False, help='The JSON file with images per sub-directory.')
    parser.add_argument("--graph", help="graph/model to be executed", required=True)
    parser.add_argument("--info", help="name of json file with model info (dimensions, layers); overrides input_height/input_width/labels/input_layer/output_layer options", default=None)
    parser.add_argument("--labels", help="name of file containing labels", required=False)
    parser.add_argument("--input_height", type=int, help="input height", default=299)
    parser.add_argument("--input_width", type=int, help="input width", default=299)
    parser.add_argument("--input_layer", help="name of input layer", default="Placeholder")
    parser.add_argument("--output_layer", help="name of output layer", default="final_result")
    parser.add_argument("--input_mean", type=int, help="input mean", default=0)
    parser.add_argument("--input_std", type=int, help="input std", default=255)
    parser.add_argument('--output_preds', type=str, required=True, help='The CSV file to store the predictions in.')
    parser.add_argument('--output_stats', type=str, required=True, help='The CSV file to store the statistics in.')
    parser.add_argument('--logging_verbosity', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'], help='How much logging output should be produced.')
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

    graph = load_graph(args.graph)

    with tf.compat.v1.Session(graph=graph) as sess:
        generate_stats(sess, graph, input_layer, output_layer, labels, args.image_dir, args.image_list,
                 input_height, input_width, args.input_mean, args.input_std,
                 args.output_preds, args.output_stats, args.logging_verbosity)


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
