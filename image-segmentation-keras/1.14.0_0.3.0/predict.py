import argparse
from datetime import datetime
import os
import time
import traceback
from keras_segmentation.predict import model_from_checkpoint_path
from image_complete import auto

SUPPORTED_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]
""" supported file extensions (lower case). """

MAX_INCOMPLETE = 3
""" the maximum number of times an image can return 'incomplete' status before getting moved/deleted. """


def predict_on_images(model, input_dir, output_dir, tmp_dir, delete_input, clash_suffix="-in"):
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
                model.predict_segmentation(inp=im_list[i], out_fname=out_file)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', help='Directory with checkpoint file(s) and _config.json (checkpoint names: ".X" with X=0..n)', required=True, default=None)
    parser.add_argument('--prediction_in', help='Path to the test images', required=True, default=None)
    parser.add_argument('--prediction_out', help='Path to the output csv files folder', required=True, default=None)
    parser.add_argument('--prediction_tmp', help='Path to the temporary csv files folder', required=False, default=None)
    parser.add_argument('--continuous', action='store_true', help='Whether to continuously load test images and perform prediction', required=False, default=False)
    parser.add_argument('--delete_input', action='store_true', help='Whether to delete the input images rather than move them to --prediction_out directory', required=False, default=False)
    parser.add_argument('--clash_suffix', help='The file name suffix to use in case the input file is already a PNG and moving it to the output directory would overwrite the prediction PNG', required=False, default="-in")
    parsed = parser.parse_args()

    try:
        model_dir = os.path.join(parsed.checkpoints, '')
        print("Loading model from %s" % model_dir)
        model = model_from_checkpoint_path(model_dir)
        while True:
            predict_on_images(model, parsed.prediction_in, parsed.prediction_out, parsed.prediction_tmp,
                              parsed.delete_input, clash_suffix=parsed.clash_suffix)
            if not parsed.continuous:
                break

    except Exception as e:
        print(traceback.format_exc())
