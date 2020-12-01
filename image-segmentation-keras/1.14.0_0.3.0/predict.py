import argparse
from datetime import datetime
import os
import time
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

SUPPORTED_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]
""" supported file extensions (lower case). """

MAX_INCOMPLETE = 3
""" the maximum number of times an image can return 'incomplete' status before getting moved/deleted. """


def predict(model, inp, out_fname=None, verbose=False):
    """
    Generates a prediction with the model.

    :param model: the model to use
    :param inp: the image, either a numpy array or a filename
    :type inp: np.ndarray or str
    :param out_fname: the name for the mask file to generate
    :type out_fname: str
    :param verbose: whether to output more logging information
    :type verbose: bool
    :return: the tuple of prediction and mask arrays (np.ndarray)
    :rtype: tuple
    """

    assert (inp is not None)
    assert ((type(inp) is np.ndarray) or isinstance(inp, six.string_types)),\
        "Input should be the CV image or the input file name"

    if isinstance(inp, six.string_types):
        inp = cv2.imread(inp)

    assert len(inp.shape) == 3, "Image should be h,w,3 "

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
        im = Image.fromarray(pr_mask)
        im = im.convert("P")
        im.putpalette(class_colors)
        im.save(out_fname)

    return pr, pr_mask


def predict_on_images(model, input_dir, output_dir, tmp_dir, delete_input, clash_suffix="-in", verbose=False):
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
    :param clash_suffix: the suffix to use for clashes, ie when the input is already a PNG image
    :type clash_suffix: str
    :param verbose: whether to output more logging information
    :type verbose: bool
    """

    # counter for keeping track of images that cannot be processed
    incomplete_counter = dict()
    num_imgs = 1

    while True:
        start_time = datetime.now()
        im_list = []
        # Loop to pick up images equal to num_imgs or the remaining images if less
        for image_path in os.listdir(input_dir):
            # Load images only
            ext_lower = os.path.splitext(image_path)[1]
            if ext_lower in SUPPORTED_EXTS:
                full_path = os.path.join(input_dir, image_path)
                if auto.is_image_complete(full_path):
                    im_list.append(full_path)
                else:
                    if not full_path in incomplete_counter:
                        incomplete_counter[full_path] = 1
                    else:
                        incomplete_counter[full_path] = incomplete_counter[full_path] + 1

            # remove images that cannot be processed
            remove_from_blacklist = []
            for k in incomplete_counter:
                if incomplete_counter[k] == MAX_INCOMPLETE:
                    print("%s - %s" % (str(datetime.now()), os.path.basename(k)))
                    remove_from_blacklist.append(k)
                    try:
                        if delete_input:
                            print("  flagged as incomplete {} times, deleting\n".format(MAX_INCOMPLETE))
                            os.remove(k)
                        else:
                            print("  flagged as incomplete {} times, skipping\n".format(MAX_INCOMPLETE))
                            os.rename(k, os.path.join(output_dir, os.path.basename(k)))
                    except:
                        print(traceback.format_exc())

            for k in remove_from_blacklist:
                del incomplete_counter[k]

            if len(im_list) == num_imgs:
                break

        if len(im_list) == 0:
            time.sleep(1)
            break
        else:
            print("%s - %s" % (str(datetime.now()), ", ".join(os.path.basename(x) for x in im_list)))

        try:
            for i in range(len(im_list)):
                parts = os.path.splitext(os.path.basename(im_list[i]))
                if tmp_dir is not None:
                    out_file = os.path.join(tmp_dir, parts[0] + ".png")
                else:
                    out_file = os.path.join(output_dir, parts[0] + ".png")
                #model.predict_segmentation(inp=im_list[i], out_fname=out_file)
                predict(model, im_list[i], out_fname=out_file, verbose=verbose)
        except:
            print("Failed processing images: {}".format(",".join(im_list)))
            print(traceback.format_exc())

        # Move finished images to output_path or delete it
        for i in range(len(im_list)):
            if delete_input:
                os.remove(im_list[i])
            else:
                # PNG input clashes with output, append suffix
                if im_list[i].lower().endswith(".png"):
                    parts = os.path.splitext(os.path.basename(im_list[i]))
                    os.rename(im_list[i], os.path.join(output_dir, parts[0] + clash_suffix + parts[1]))
                else:
                    os.rename(im_list[i], os.path.join(output_dir, os.path.basename(im_list[i])))

        end_time = datetime.now()
        inference_time = end_time - start_time
        inference_time = int(inference_time.total_seconds() * 1000)
        print("  Inference + I/O time: {} ms\n".format(inference_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Allows continuous processing of images appearing in the input directory, storing the predictions "
                    + "in the output directory. Input files can either be moved to the output directory or deleted.")
    parser.add_argument('--checkpoints_path', help='Directory with checkpoint file(s) and _config.json (checkpoint names: ".X" with X=0..n)', required=True, default=None)
    parser.add_argument('--prediction_in', help='Path to the test images', required=True, default=None)
    parser.add_argument('--prediction_out', help='Path to the output csv files folder', required=True, default=None)
    parser.add_argument('--prediction_tmp', help='Path to the temporary csv files folder', required=False, default=None)
    parser.add_argument('--continuous', action='store_true', help='Whether to continuously load test images and perform prediction', required=False, default=False)
    parser.add_argument('--delete_input', action='store_true', help='Whether to delete the input images rather than move them to --prediction_out directory', required=False, default=False)
    parser.add_argument('--clash_suffix', help='The file name suffix to use in case the input file is already a PNG and moving it to the output directory would overwrite the prediction PNG', required=False, default="-in")
    parser.add_argument('--memory_fraction', type=float, help='Memory fraction to use by tensorflow, i.e., limiting memory usage', required=False, default=0.5)
    parser.add_argument('--verbose', action='store_true', help='Whether to output more logging info', required=False, default=False)
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

        # predict
        while True:
            predict_on_images(model, parsed.prediction_in, parsed.prediction_out, parsed.prediction_tmp,
                              parsed.delete_input, clash_suffix=parsed.clash_suffix, verbose=parsed.verbose)
            if not parsed.continuous:
                break

    except Exception as e:
        print(traceback.format_exc())
