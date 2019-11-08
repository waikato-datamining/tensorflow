# Original Code from
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Licensed under Apache 2.0 (Most likely, since tensorflow is Apache 2.0)
# Modifications Copyright (C) 2018 University of Waikato, Hamilton, NZ
#
# Performs predictions on combined images from all images present in the folder passed as "prediction_in", then outputs the results as
# csv files in the folder passed as "prediction_out"
#
# The number of images to combine at a time before prediction can be specified (passed as a parameter)
# Score threshold can be specified (passed as a parameter) to ignore all rois with low score
# Can run in a continuous mode, where it will run indefinitely

import numpy as np
import os
import sys
import tensorflow as tf
import argparse
from PIL import Image
from tensorflow import Graph
from datetime import datetime
import time

sys.path.append("..")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util

OUTPUT_COMBINED = False
""" Whether to output CSV file with ROIs for combined images as well (only for debugging). """


def load_image_into_numpy_array(image):
    """
    Method to convert the image into a numpy array.
    faster solution via np.fromstring found here:
    https://stackoverflow.com/a/42036542/4698227

    :param image: the image object to convert
    :type image: Image
    :return: the numpy array
    :rtype: nd.array
    """

    im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((image.size[1], image.size[0], 3))
    return im_arr


def run_inference_for_single_image(image, sess):
    """
    Obtain predictions for image.

    :param image: the image to generate predictions for
    :type image: str
    :param sess: the tensorflow session
    :type sess: tf.Session
    :return: the predictions
    :rtype: dict
    """

    # Get handles to input and output tensors
    ops = tf.compat.v1.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
                tensor_name)
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                           feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def load_frozen_graph(graph_path):
    """
    Loads the provided frozen graph into detection_graph to use with prediction.

    :param graph_path: the path to the frozen graph
    :type graph_path: str
    :return: the graph
    :rtype: tf.Graph
    """

    graph = tf.Graph()  # type: Graph
    with graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph


def remove_alpha_channel(image):
    """
    Converts the Image object to RGB.

    :param image: the image object to convert if necessary
    :type image: Image
    :return: the converted object
    :rtype: Image
    """
    if image.mode is 'RGBA' or 'ARGB':
        return image.convert('RGB')
    else:
        return image


def predict_on_images(input_dir, sess, output_dir, score_threshold, categories, num_imgs, inference_times, delete_input):
    """
    Method performing predictions on all images ony by one or combined as specified by the int value of num_imgs.

    :param input_dir: the directory with the images
    :type input_dir: str
    :param sess: the tensorflow session
    :type sess: tf.Session
    :param output_dir: the output directory to move the images to and store the predictions
    :type output_dir: str
    :param score_threshold: the minimum score predictions have to have
    :type score_threshold: float
    :param categories: the label map
    :param num_imgs: the number of images to combine into one before presenting to graph
    :type num_imgs: int
    :param inference_times: whether to output a CSV file with the inference times
    :type inference_times: bool
    :param delete_input: whether to delete the input images rather than moving them to the output directory
    :type delete_input: bool
    """

    # Iterate through all files present in "test_images_directory"
    total_time = 0
    times = list()
    times.append("Image(s)_file_name(s),Total_time(ms),Number_of_images,Time_per_image(ms)\n")
    while True:
        start_time = datetime.now()
        im_list = []
        # Loop to pick up images equal to num_imgs or the remaining images if less
        for image_path in os.listdir(input_dir):
            # Load images only, currently supporting only jpg and png
            # TODO image complete?
            if image_path.lower().endswith(".jpg") or image_path.lower().endswith(".png"):
                im_list.append(os.path.join(input_dir, image_path))
            if len(im_list) == num_imgs:
                break

        if len(im_list) == 0:
            time.sleep(1)
            break
        else:
            print("%s - %s" % (str(datetime.now()), ", ".join(os.path.basename(x) for x in im_list)))

        # Combining picked up images
        i = len(im_list)
        combined = []
        comb_img = None
        if i > 1:
            while i != 0:
                if comb_img is None:
                    img2 = Image.open(im_list[i-1])
                    img1 = Image.open(im_list[i-2])
                    i -= 1
                    combined.append(os.path.join(output_dir, "combined.png"))
                else:
                    img2 = comb_img
                    img1 = Image.open(im_list[i-1])
                i -= 1
                # Remove alpha channel if present
                img1 = remove_alpha_channel(img1)
                img2 = remove_alpha_channel(img2)
                w1, h1 = img1.size
                w2, h2 = img2.size
                comb_img = np.zeros((h1+h2, max(w1, w2), 3), np.uint8)
                comb_img[:h1, :w1, :3] = img1
                comb_img[h1:h1+h2, :w2, :3] = img2
                comb_img = Image.fromarray(comb_img)

        if comb_img is None:
            im_name = im_list[0]
            image = Image.open(im_name)
            image = remove_alpha_channel(image)
        else:
            im_name = combined[0]
            image = remove_alpha_channel(comb_img)

        image_np = load_image_into_numpy_array(image)
        output_dict = run_inference_for_single_image(image_np, sess)

        # Loading results
        boxes = output_dict['detection_boxes']
        scores = output_dict['detection_scores']
        classes = output_dict['detection_classes']

        if OUTPUT_COMBINED:
            roi_path = "{}/{}-rois-combined.csv".format(output_dir, os.path.splitext(os.path.basename(im_name))[0])
            with open(roi_path, "w") as roi_file:
                # File header
                roi_file.write("file,x0,y0,x1,y1,x0n,y0n,x1n,y1n,label,label_str,score\n")
                for index in range(output_dict['num_detections']):
                    y0n, x0n, y1n, x1n = boxes[index]
                    label = classes[index]
                    label_str = categories[label - 1]['name']
                    score = scores[index]

                    # Ignore this roi if the score is less than the provided threshold
                    if score < score_threshold:
                        continue

                    # Translate roi coordinates into image coordinates
                    x0 = x0n * image.width
                    y0 = y0n * image.height
                    x1 = x1n * image.width
                    y1 = y1n * image.height

                    roi_file.write(
                        "{},{},{},{},{},{},{},{},{},{},{},{}\n".format(os.path.basename(im_name), x0, y0, x1, y1,
                                                                       x0n, y0n, x1n, y1n, label, label_str, score))

        # Code for splitting rois to multiple csv's, one csv per image before combining
        max_height = 0
        prev_min = 0
        for i in range(len(im_list)):
            img = Image.open(im_list[i])
            img_height = img.height
            min_height = prev_min
            max_height += img_height
            prev_min = max_height
            roi_path = "{}/{}-rois.csv".format(output_dir, os.path.splitext(os.path.basename(im_list[i]))[0])
            roi_path_tmp = "{}/{}-rois.tmp".format(output_dir, os.path.splitext(os.path.basename(im_list[i]))[0])
            with open(roi_path_tmp, "w") as roi_file:
                # File header
                roi_file.write("file,x0,y0,x1,y1,x0n,y0n,x1n,y1n,label,label_str,score\n")
                # rois
                for index in range(output_dict['num_detections']):
                    y0n, x0n, y1n, x1n = boxes[index]
                    label = classes[index]
                    label_str = categories[label - 1]['name']
                    score = scores[index]

                    # Ignore this roi if the score is less than the provided threshold
                    if score < score_threshold:
                        continue

                    # Translate roi coordinates into combined image coordinates
                    x0 = x0n * image.width
                    y0 = y0n * image.height
                    x1 = x1n * image.width
                    y1 = y1n * image.height

                    if y0 > max_height or y1 > max_height:
                        continue
                    elif y0 < min_height or y1 < min_height:
                        continue

                    # output
                    roi_file.write("{},{},{},{},{},{},{},{},{},{},{},{}\n".format(os.path.basename(im_name),
                                                                                  x0, y0, x1, y1, x0n, y0n, x1n, y1n,
                                                                                  label, label_str, score))
            os.rename(roi_path_tmp, roi_path)

        # Move finished images to output_path or delete it
        for i in range(len(im_list)):
            if delete_input:
                os.remove(im_list[i])
            else:
                os.rename(im_list[i], os.path.join(output_dir, os.path.basename(im_list[i])))

        end_time = datetime.now()
        inference_time = end_time - start_time
        inference_time = int(inference_time.total_seconds() * 1000)
        time_per_image = int(inference_time / len(im_list))
        if inference_times:
            l = ""
            for i in range(len(im_list)):
                l += ("{}|".format(os.path.basename(im_list[i])))
            l += ",{},{},{}\n".format(inference_time, len(im_list), time_per_image)
            times.append(l)
        print("  Inference + I/O time: {} ms\n".format(inference_time))
        total_time += inference_time

    if inference_times:
        with open(os.path.join(output_dir, "inference_time.csv"), "w") as time_file:
            for l in times:
                time_file.write(l)
        with open(os.path.join(output_dir, "total_time.txt"), "w") as total_time_file:
            total_time_file.write("Total inference and I/O time: {} ms\n".format(total_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', help='Path to the frozen detection graph', required=True, default=None)
    parser.add_argument('--labels', help='Path to the labels map', required=True, default=None)
    parser.add_argument('--prediction_in', help='Path to the test images', required=True, default=None)
    parser.add_argument('--prediction_out', help='Path to the output csv files folder', required=True, default=None)
    parser.add_argument('--score', type=float, help='Score threshold to include in csv file', required=False, default=0.0)
    parser.add_argument('--num_classes', type=int, help='Number of classes', required=True, default=2)
    parser.add_argument('--num_imgs', type=int, help='Number of images to combine', required=False, default=1)
    parser.add_argument('--status', help='file path for predict exit status file', required=False, default=None)
    parser.add_argument('--continuous', action='store_true', help='Whether to continuously load test images and perform prediction', required=False, default=False)
    parser.add_argument('--output_inference_time', action='store_true', help='Whether to output a CSV file with inference times in the --prediction_output directory', required=False, default=False)
    parser.add_argument('--delete_input', action='store_true', help='Whether to delete the input images rather than move them to --prediction_out directory', required=False, default=False)
    parsed = parser.parse_args()

    try:
        # Path to frozen detection graph. This is the actual model that is used for the object detection
        detection_graph = load_frozen_graph(parsed.graph)

        # Getting classes strings from the label map
        label_map = label_map_util.load_labelmap(parsed.labels)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=parsed.num_classes,
                                                                    use_display_name=True)

        with detection_graph.as_default():
            with tf.compat.v1.Session() as sess:
                while True:
                    # Performing the prediction and producing the csv files
                    predict_on_images(parsed.prediction_in, sess, parsed.prediction_out, parsed.score,
                                      categories, parsed.num_imgs, parsed.output_inference_time,
                                      parsed.delete_input)

                    # Exit if not continuous
                    if not parsed.continuous:
                        break
                if parsed.status is not None:
                    with open(parsed.status, 'w') as f:
                        f.write("Success")

    except Exception as e:
        print(e)
        if parsed.status is not None:
            with open(parsed.status, 'w') as f:
                f.write(str(e))