# Original Code from
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Licensed under Apache 2.0 (Most likely, since tensorflow is Apache 2.0)
# Modifications Copyright (C) 2018-2020 University of Waikato, Hamilton, NZ
#
# Performs predictions on combined images from all images present in the folder passed as "prediction_in", then outputs the results as
# csv files in the folder passed as "prediction_out"
#
# The number of images to combine at a time before prediction can be specified (passed as a parameter)
# Score threshold can be specified (passed as a parameter) to ignore all rois with low score
# Can run in a continuous mode, where it will run indefinitely

import sys
import numpy as np
import tensorflow as tf

sys.path.append("..")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util


def load_labels(label_map_path, num_classes=None, use_display_name=True):
    """
    Loads the labels (map and categories) from the protobuf text file.

    :param label_map_path: the protobuf text file with the labels
    :type label_map_path: str
    :param num_classes: the maximum number of classes to use, use None to determine maximum from label_map_path
    :type num_classes: int
    :param use_display_name: choose whether to load 'display_name' field as category name.
           If False or if the display_name field does not exist, uses 'name' field as category names instead.
    :type use_display_name: bool
    :return: the label map and the categories as tuple
    :rtype: tuple
    """

    label_map = label_map_util.load_labelmap(label_map_path)
    if num_classes is None:
        num_classes = len(label_map.item)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=num_classes, use_display_name=use_display_name)
    return label_map, categories


def inference_for_image(image, graph, sess):
    """
    Obtain predictions for image.

    :param image: the image to generate predictions for
    :type image: str
    :param graph: the graph to use
    :type graph: tf.Graph()
    :param sess: the tensorflow session
    :type sess: tf.Session
    :return: the predictions
    :rtype: dict
    """

    # Get handles to input and output tensors
    ops = graph.get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = graph.get_tensor_by_name(tensor_name)
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
        tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
    image_tensor = graph.get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                           feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]

    return output_dict
