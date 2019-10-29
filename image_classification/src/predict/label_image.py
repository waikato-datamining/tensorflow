# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

from utils.prediction_utils import load_graph, load_labels, read_tensor_from_image_file, tensor_to_probs, top_k_probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="image to be processed", required=True)
    parser.add_argument("--graph", help="graph/model to be executed", required=True)
    parser.add_argument("--labels", help="name of file containing labels", required=True)
    parser.add_argument("--input_height", type=int, help="input height", default=299)
    parser.add_argument("--input_width", type=int, help="input width", default=299)
    parser.add_argument("--input_mean", type=int, help="input mean", default=0)
    parser.add_argument("--input_std", type=int, help="input std", default=255)
    parser.add_argument("--input_layer", help="name of input layer", default="Placeholder")
    parser.add_argument("--output_layer", help="name of output layer", default="final_result")
    parser.add_argument("--top_x", type=int, help="output only the top K labels; use <1 for all", default=5)
    args = parser.parse_args()

    graph = load_graph(args.graph)
    labels = load_labels(args.labels)
    tensor = read_tensor_from_image_file(
        args.image,
        input_height=args.input_height,
        input_width=args.input_width,
        input_mean=args.input_mean,
        input_std=args.input_std)

    results = tensor_to_probs(graph, args.input_layer, args.output_layer, tensor)
    top_x = top_k_probs(results, args.top_x)
    if args.top_x > 0:
        print("Top " + str(args.top_x) + " labels")
    else:
        print("All labels")
    for i in top_x:
        print("- " + labels[i] + ":", results[i])
