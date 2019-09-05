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

sys.path.append("..")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util

# Method to convert the image into a numpy array
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

# This is where the actual prediction occur
def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
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
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

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

# Loads the provided frozen graph into detection_graph to use with prediction
def load_frozen_graph(frozen_graph_path):
    detection_graph = tf.Graph()  # type: Graph
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

# Method performing predictions on all images ony by one or combined as specified by the int value of num_imgs
def predict_on_images(test_images_directory, detection_graph, output_path, score_threshold, categories, file, num_imgs):
    image_no = 1    # A variable used only to print out the progress
    time_delta_total = 0
    images_total = 0
    # Iterate through all files present in "test_images_directory"
    while True:
        im_list = []
        # Loop to pick up images equal to num_imgs or the remaining images if less
        for image_path in os.listdir(test_images_directory):
            print("Processing image {}\n".format(image_no))
            # Load images only, currently supporting only jpg and png
            if image_path.lower().endswith(".jpg") or image_path.lower().endswith(".png"):
                im_list.append(os.path.join(test_images_directory, image_path))
                image_no += 1
            if len(im_list) == num_imgs:
                break
        
        if len(im_list) == 0:
            break
        
        # Combining picked up images
        i = len(im_list)
        combined = []
        if i > 1:
            comb_file = os.path.join(output_path, "combined.png")
            while i != 0:
                if len(combined) == 0:
                    img2 = Image.open(im_list[i-1])
                    img1 = Image.open(im_list[i-2])
                    i -= 1
                    combined.append(comb_file)
                else:
                    img2 = Image.open(combined[0])
                    img1 = Image.open(im_list[i-1])
                i -= 1
                # Remove alpha channel if present
                if img1.mode is 'RGBA' or 'ARGB':
                    img1 = img1.convert('RGB')
                if img2.mode is 'RGBA' or 'ARGB':
                    img2 = img2.convert('RGB')
                w1, h1 = img1.size
                w2, h2 = img2.size
                comb_img = np.zeros((h1+h2, max(w1, w2), 3), np.uint8)
                comb_img[:h1, :w1, :3] = img1
                comb_img[h1:h1+h2, :w2, :3] = img2
                comb_img = Image.fromarray(comb_img)
                comb_img.save(comb_file)
                
        if len(combined) == 0:
            im_name = im_list[0]
        else:
            im_name = combined[0]
            
        image = Image.open(im_name)

        # Remove alpha channel if present
        if im_name.lower().endswith(".png") and (image.mode is 'RGBA' or 'ARGB'):
            image = image.convert('RGB')
        # Convert the image into a numpy array
        image_np = load_image_into_numpy_array(image)
        # Record current time
        time_start = datetime.now()
        # Actual detection
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        # Compute time needed to predict per image in ms
        time_end = datetime.now()
        time_delta = time_end - time_start
        if image_no != 2:
            time_delta_total += int(time_delta.total_seconds() * 1000)
            images_total += 1

        # Loading results
        boxes = output_dict['detection_boxes']
        scores = output_dict['detection_scores']
        classes = output_dict['detection_classes']

        # Original code for writing results into a csv file for the combined image, storing it in "output_path" directory (kept to verify csv's after splitting)
        roi_path = "{}/{}-rois.csv".format(output_path, os.path.splitext(os.path.basename(im_name))[0])
        with open(roi_path, "w") as roi_file:
            # File header
            roi_file.write("file,x0,y0,x1,y1,label,label_str,score\n")
            for index in range(output_dict['num_detections']):
                y0, x0, y1, x1 = boxes[index]
                label = classes[index]
                label_str = categories[label - 1]['name']
                score = scores[index]

                # Ignore this roi if the score is less than the provided threshold
                if score < score_threshold:
                    continue

                # Translate roi coordinates into image coordinates
                x0 = x0 * image.width
                y0 = y0 * image.height
                x1 = x1 * image.width
                y1 = y1 * image.height

                roi_file.write(
                    "{},{},{},{},{},{},{},{}\n".format(os.path.basename(im_name), x0, y0, x1, y1, label, label_str,
                                                       score))
                                                       
        # Code for splitting rois to multiple csv's, one csv per image before combining
        min_height = 0
        max_height = 0
        prev_min = 0
        translation = 0
        for i in range(len(im_list)):
            img = Image.open(im_list[i])
            img_height = img.height
            min_height = prev_min
            max_height += img_height
            prev_min = max_height
            roi_path = "{}/{}-rois.csv".format(output_path, os.path.splitext(os.path.basename(im_list[i]))[0])
            with open(roi_path, "w") as roi_file:
                # File header
                roi_file.write("file,x0,y0,x1,y1,label,label_str,score\n")
                # rois
                for index in range(output_dict['num_detections']):
                    y0, x0, y1, x1 = boxes[index]
                    label = classes[index]
                    label_str = categories[label - 1]['name']
                    score = scores[index]

                    # Ignore this roi if the score is less than the provided threshold
                    if score < score_threshold:
                        continue
                        
                    # Translate roi coordinates into combined image coordinates
                    x0 = x0 * image.width
                    y0 = y0 * image.height
                    x1 = x1 * image.width
                    y1 = y1 * image.height
                    
                    if y0 > max_height or y1 > max_height:
                        continue
                    elif y0 < min_height or y1 < min_height:
                        continue

                    # output
                    roi_file.write("{},{},{},{},{},{},{},{}\n".format(os.path.basename(im_list[i]), x0, y0-translation, x1, y1-translation,
                            label, label_str, score))
            translation += img_height
        
        # Move finished images to output_path
        for i in range(len(im_list)):
            os.rename(im_list[i], os.path.join(output_path, os.path.basename(im_list[i])))
    # Save average elapsed time to file
    file.write(str(int(time_delta_total / images_total)) + " ms")

if __name__ == '__main__':

    # Arguments to be provided by the user
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', help='Path to the frozen detection graph', required=True, default=None)
    parser.add_argument('--labels', help='Path to the labels map', required=True, default=None)
    parser.add_argument('--prediction_in', help='Path to the test images', required=True, default=None)
    parser.add_argument('--prediction_out', help='Path to the output csv files folder', required=True, default=None)
    parser.add_argument('--score', type=float, help='Score threshold to include in csv file', required=False, default=0.0)
    parser.add_argument('--num_classes', type=int, help='Number of classes', required=True, default=2)
    parser.add_argument('--num_imgs', type=int, help='Number of images to combine', required=False, default=1)
    parser.add_argument('--continuous', action='store_true', help='Whether to continuously load test images and perform prediction', required=False, default=False)
    args = vars(parser.parse_args())

    # Path to frozen detection graph. This is the actual model that is used for the object detection
    frozen_graph_path = args['graph']
    detection_graph = load_frozen_graph(frozen_graph_path)

    # List of the strings that is used to add correct label for each box
    labels_map_path = args['labels']

    # The threshold of including the detected roi in the output csv
    score_threshold = args['score']

    # Getting classes strings from the label map
    num_classes = args['num_classes']
    label_map = label_map_util.load_labelmap(labels_map_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes,
                                                                use_display_name=True)

    # Loading the path to images to perform prediction on
    test_images_directory = args['prediction_in']

    # Output directory for the csv files
    output_path = args['prediction_out']

    # File to save prediction time in
    f = open(output_path + "/prediction_time.txt", "w")
    
    # Number of images to combine
    num_imgs = args['num_imgs']

    while True:
        # Performing the prediction and producing the csv files
        predict_on_images(test_images_directory, detection_graph, output_path, score_threshold, categories, f, num_imgs)

        # Exit if not continuous
        if not args['continuous']:
            break
    f.close()