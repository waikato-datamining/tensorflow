# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
# Copyright 2022 University of Waikato, Hamilton, NZ.
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

import io
import json
import traceback

from datetime import datetime
from collections import OrderedDict

from wai.tfimageclass.utils.prediction_utils import tflite_load_model, load_labels, \
    tflite_read_tensor_from_bytes, tflite_tensor_to_probs, tflite_top_k_probs, load_info_file
from rdh import Container, MessageContainer, create_parser, configure_redis, run_harness, log


def process_image(msg_cont):
    """
    Processes the message container, loading the image from the message and forwarding the object detection predictions.

    :param msg_cont: the message container to process
    :type msg_cont: MessageContainer
    """
    config = msg_cont.params.config

    try:
        start_time = datetime.now()

        image = io.BytesIO(msg_cont.message['data']).getvalue()
        tensor = tflite_read_tensor_from_bytes(image, config.input_height, config.input_width,
                                                    input_mean=config.input_mean, input_std=config.input_std)
        results = tflite_tensor_to_probs(config.interpreter, tensor)
        top_x = tflite_top_k_probs(results, config.top_x)
        predictions = OrderedDict()
        for i in range(len(top_x)):
            predictions[config.labels[top_x[i]]] = float(results[0][top_x[i]])

        msg_cont.params.redis.publish(msg_cont.params.channel_out, json.dumps(predictions))

        if config.verbose:
            log("process_images - predicted image published: %s" % msg_cont.params.channel_out)
            end_time = datetime.now()
            processing_time = end_time - start_time
            processing_time = int(processing_time.total_seconds() * 1000)
            log("process_images - finished processing image: %d ms" % processing_time)
    except KeyboardInterrupt:
        msg_cont.params.stopped = True
    except:
        log("process_images - failed to process: %s" % traceback.format_exc())


def main(parsed=None):
    """
    The main method for parsing command-line arguments and labeling.

    :param parsed: the commandline arguments, uses sys.argv if not supplied
    :type parsed: list
    """
    parser = create_parser("Uses a tflite image classification model to make predictions on images received via a Redis channel and broadcasts the predictions via another Redis channel.",
                           prog="tfic-label-redis", prefix="redis_")
    parser.add_argument("--model", help="model to be executed", required=True)
    parser.add_argument("--info", help="name of json file with model info (dimensions, layers); overrides input_height/input_width/labels/input_layer/output_layer options", default=None)
    parser.add_argument("--labels", help="name of file containing labels", required=False)
    parser.add_argument("--input_height", type=int, help="input height", default=299)
    parser.add_argument("--input_width", type=int, help="input width", default=299)
    parser.add_argument("--input_layer", help="name of input layer", default="Placeholder")
    parser.add_argument("--output_layer", help="name of output layer", default="final_result")
    parser.add_argument("--input_mean", type=int, help="input mean", default=0)
    parser.add_argument("--input_std", type=int, help="input std", default=255)
    parser.add_argument("--top_x", type=int, help="output only the top K labels; use <1 for all", default=5)
    parser.add_argument("--verbose", action="store_true", help="whether to output some debugging information")
    parsed = parser.parse_args(args=parsed)

    # values from options
    labels = None
    input_height = parsed.input_height
    input_width = parsed.input_width
    input_layer = parsed.input_layer
    output_layer = parsed.output_layer

    # override from info file?
    if parsed.info is not None:
        input_height, input_width, input_layer, output_layer, labels = load_info_file(parsed.info)

    if (labels is None) and (parsed.labels is not None):
        labels = load_labels(parsed.labels)
    if labels is None:
        raise Exception("No labels determined, either supply --info or --labels!")

    config = Container()
    config.labels = labels
    config.input_mean = parsed.input_mean
    config.input_std = parsed.input_std
    config.input_height = input_height
    config.input_width = input_width
    config.input_layer = input_layer
    config.output_layer = output_layer
    config.top_x = parsed.top_x
    config.interpreter = tflite_load_model(parsed.model)
    config.verbose = parsed.verbose

    params = configure_redis(parsed, config=config)
    run_harness(params, process_image)


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
