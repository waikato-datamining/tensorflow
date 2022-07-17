import argparse
import os
import traceback
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras_segmentation.predict import model_from_checkpoint_path
from image_complete import auto
import six
import numpy as np
import cv2
from keras_segmentation.data_utils.data_loader import get_image_array
from keras_segmentation.models.config import IMAGE_ORDERING
from PIL import Image
import random
from sfp import Poller

# colors taken from:
# conservative 8-color palettes for color blindness
# http://mkweb.bcgsc.ca/colorblind/palettes.mhtml
class_colors = [
      0,   0,   0,
     34, 113, 178,
     61, 183, 233,
    247,  72, 165,
     53, 155, 115,
    213,  94,   0,
    230, 159,   0,
    240, 228,  66,
]
num_colors = 256 - 8
r = [random.randint(0,255) for _ in range(num_colors)]
g = [random.randint(0,255) for _ in range(num_colors)]
b = [random.randint(0,255) for _ in range(num_colors)]
for rgb in zip(r, g, b):
    class_colors.extend(rgb)

SUPPORTED_EXTS = [".jpg", ".jpeg"]
""" supported file extensions (lower case with dot). """


def predict(model, inp, out_fname=None, colors=None, remove_background=False, verbose=False):
    """
    Generates a prediction with the model.

    :param model: the model to use
    :param inp: the image, either a numpy array or a filename
    :type inp: np.ndarray or str
    :param out_fname: the name for the mask file to generate
    :type out_fname: str
    :param colors: the list of colors to use (flat list of r,g,b values, eg "0, 0, 0, 127, 127, 127, ..."), needs to have 768 entries
    :type colors: list
    :param remove_background: whether to use the mask to remove the background
    :type remove_background: bool
    :param verbose: whether to output more logging information
    :type verbose: bool
    :return: the tuple of prediction and mask arrays (np.ndarray)
    :rtype: tuple
    """

    with graph.as_default():
        assert (inp is not None)
        assert ((type(inp) is np.ndarray) or isinstance(inp, six.string_types)),\
            "Input should be the CV image or the input file name"

        if colors is not None:
            assert (len(colors) == 768), "list of colors must be 768 (256 r,g,b triplets)"

        if isinstance(inp, six.string_types):
            inp = cv2.imread(inp)

        assert len(inp.shape) == 3, "Image should be h,w,3 "

        if remove_background:
            inp_orig = np.copy(inp)

        output_width = model.output_width
        output_height = model.output_height
        input_width = model.input_width
        input_height = model.input_height
        n_classes = model.n_classes

        x = get_image_array(inp, input_width, input_height,
                            ordering=IMAGE_ORDERING)
        pr = model.predict(np.array([x]))[0]
        pr = pr.reshape((output_height,  output_width, n_classes)).argmax(axis=2)

        original_h = inp.shape[0]
        original_w = inp.shape[1]
        pr_mask = pr.astype('uint8')
        pr_mask = cv2.resize(pr_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        if verbose:
            unique, count = np.unique(pr_mask, return_counts=True)
            print("  unique:", unique)
            print("  count:", count)

        if out_fname is not None:
            if remove_background:
                onebit_mask = np.where(pr_mask > 0, 1, 0)
                no_background = inp_orig.copy()
                for i in range(inp_orig.shape[2]):
                    no_background[:,:,i] = no_background[:,:,i] * onebit_mask
                cv2.imwrite(out_fname, no_background)
            else:
                im = Image.fromarray(pr_mask)
                im = im.convert("P")
                if colors is not None:
                    im.putpalette(colors)
                else:
                    im.putpalette(class_colors)
                im.save(out_fname)

        return pr, pr_mask


def check_image(fname, poller):
    """
    Check method that ensures the image is valid.

    :param fname: the file to check
    :type fname: str
    :param poller: the Poller instance that called the method
    :type poller: Poller
    :return: True if complete
    :rtype: bool
    """
    result = auto.is_image_complete(fname)
    poller.debug("Image complete:", fname, "->", result)
    return result


def process_image(fname, output_dir, poller):
    """
    Method for processing an image.

    :param fname: the image to process
    :type fname: str
    :param output_dir: the directory to write the image to
    :type output_dir: str
    :param poller: the Poller instance that called the method
    :type poller: Poller
    :return: the list of generated output files
    :rtype: list
    """
    result = []
    try:
        out_file = os.path.join(output_dir, os.path.splitext(os.path.basename(fname))[0] + ".png")
        predict(poller.params.model, fname, out_fname=out_file, colors=poller.params.colors,
                remove_background=poller.params.remove_background, verbose=poller.verbose)
        result.append(out_file)
    except KeyboardInterrupt:
        poller.keyboard_interrupt()
    except:
        poller.error("Failed to process image: %s\n%s" % (fname, traceback.format_exc()))
    return result


def predict_on_images(model, input_dir, output_dir, tmp_dir, delete_input,
                      colors=None, continuous=False, poll_wait=1.0,
                      use_watchdog=False, watchdog_check_interval=10.0,
                      remove_background=False, verbose=False, quiet=False):
    """
    Performs predictions on images found in input_dir and outputs the prediction PNG files in output_dir.

    :param model: the model to use
    :param input_dir: the directory with the images
    :type input_dir: str
    :param output_dir: the output directory to move the images to and store the predictions
    :type output_dir: str
    :param tmp_dir: the temporary directory to store the predictions until finished
    :type tmp_dir: str
    :param delete_input: whether to delete the input images rather than moving them to the output directory
    :type delete_input: bool
    :param colors: the list of colors to use (flat list of r,g,b values, eg "0, 0, 0, 127, 127, 127"), will get padded to 256 triplets
    :type colors: list
    :param poll_wait: the amount of seconds between polls when not in watchdog mode
    :type poll_wait: float
    :param continuous: whether to poll for files continuously
    :type continuous: bool
    :param use_watchdog: whether to react to file creation events rather than use fixed-interval polling
    :type use_watchdog: bool
    :param watchdog_check_interval: the interval for the watchdog process to check for files that were missed due to potential race conditions
    :type watchdog_check_interval: float
    :param remove_background: whether to use the mask to remove the background
    :type remove_background: bool
    :param verbose: whether to output more logging information
    :type verbose: bool
    :param quiet: whether to suppress output
    :type quiet: bool
    """

    # ensure colors list has correct length for PNG palette
    if colors is not None:
        if len(colors) < 768:
            colors.extend(class_colors[len(colors):])
        if len(colors) > 768:
            colors = colors[0:768]

    poller = Poller()
    poller.input_dir = input_dir
    poller.output_dir = output_dir
    poller.tmp_dir = tmp_dir
    poller.extensions = SUPPORTED_EXTS
    poller.delete_input = delete_input
    poller.verbose = verbose
    poller.progress = not quiet
    poller.check_file = check_image
    poller.process_file = process_image
    poller.poll_wait = poll_wait
    poller.continuous = continuous
    poller.use_watchdog = use_watchdog
    poller.watchdog_check_interval = watchdog_check_interval
    poller.params.model = model
    poller.params.colors = colors
    poller.params.remove_background = remove_background
    poller.poll()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Allows continuous processing of images appearing in the input directory, storing the predictions "
                    + "in the output directory. Input files can either be moved to the output directory or deleted.")
    parser.add_argument('--checkpoints_path', help='Directory with checkpoint file(s) and _config.json, no trailing slash (checkpoint names: ".X" with X=0..n)', required=True, default=None)
    parser.add_argument('--prediction_in', help='Path to the test images', required=True, default=None)
    parser.add_argument('--prediction_out', help='Path to the output csv files folder', required=True, default=None)
    parser.add_argument('--prediction_tmp', help='Path to the temporary csv files folder', required=False, default=None)
    parser.add_argument('--poll_wait', type=float, help='poll interval in seconds when not using watchdog mode', required=False, default=1.0)
    parser.add_argument('--continuous', action='store_true', help='Whether to continuously load test images and perform prediction', required=False, default=False)
    parser.add_argument('--use_watchdog', action='store_true', help='Whether to react to file creation events rather than performing fixed-interval polling', required=False, default=False)
    parser.add_argument('--watchdog_check_interval', type=float, help='check interval in seconds for the watchdog', required=False, default=10.0)
    parser.add_argument('--delete_input', action='store_true', help='Whether to delete the input images rather than move them to --prediction_out directory', required=False, default=False)
    parser.add_argument('--memory_fraction', type=float, help='Memory fraction to use by tensorflow, i.e., limiting memory usage', required=False, default=0.5)
    parser.add_argument('--colors', help='The list of colors (RGB triplets) to use for the PNG palette, e.g.: 0,0,0,255,0,0,0,0,255 for black,red,blue', required=False, default=None)
    parser.add_argument('--remove_background', action='store_true', help='Whether to use the predicted mask to remove the background and output this modified image instead of the mask', required=False, default=False)
    parser.add_argument('--verbose', action='store_true', help='Whether to output more logging info', required=False, default=False)
    parser.add_argument('--quiet', action='store_true', help='Whether to suppress output', required=False, default=False)
    parsed = parser.parse_args()

    try:
        # apply memory usage
        print("Using memory fraction: %f" % parsed.memory_fraction)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = parsed.memory_fraction
        set_session(tf.Session(config=config))

        # load model
        model_dir = os.path.join(parsed.checkpoints_path, '')
        print("Loading model from %s" % model_dir)
        model = model_from_checkpoint_path(model_dir)

        global graph
        graph = tf.get_default_graph()

        # color palette
        colors = []
        if parsed.colors is not None:
            colors = parsed.colors.split(",")
            colors = [int(x) for x in colors]

        # predict
        predict_on_images(model, parsed.prediction_in, parsed.prediction_out, parsed.prediction_tmp,
                          parsed.delete_input, colors=colors, poll_wait=parsed.poll_wait, continuous=parsed.continuous,
                          use_watchdog=parsed.use_watchdog, watchdog_check_interval=parsed.watchdog_check_interval,
                          remove_background=parsed.remove_background, verbose=parsed.verbose, quiet=parsed.quiet)

    except Exception as e:
        print(traceback.format_exc())
