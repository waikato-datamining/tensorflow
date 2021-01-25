# Copyright 2021 University of Waikato, Hamilton, NZ.
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
import traceback
import tensorflow as tf

def export(saved_model_dir, tflite_model):
    """
    Exports a saved Tensorflow model to a Tensorflow lite one.

    :param saved_model_dir: the saved model to convert
    :type saved_model_dir: str
    :param tflite_model: the file to write the tensorflow lite to
    :type tflite_model: str
    """

    # Convert the model.
    converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite = converter.convert()

    # Save the model.
    with open(tflite_model, 'wb') as f:
      f.write(tflite)


def main(args=None):
    """
    The main method for parsing command-line arguments and labeling.

    :param args: the commandline arguments, uses sys.argv if not supplied
    :type args: list
    """
    parser = argparse.ArgumentParser(
        description="Exports a Tensorflow model as Tensorflow lite one.",
        prog="tfic-export",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--saved_model_dir', type=str, default='', required=True, help='Path to the saved Tensorflow model directory.')
    parser.add_argument('--tflite_model', type=str, default='', required=True, help='The file to export the Tensorflow lite model to.')
    args = parser.parse_args(args=args)

    export(args.saved_model_dir, args.tflite_model)

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
