"""
Takes an image and an associated CSV file with bounding box information
and attempts to tighten up the bounding boxes by identifying the actual
objects in the original bounding boxes and then updates the bounding
box information in the CSV file.

Copyright (C) 2018 University of Waikato, Hamilton, NZ
Licensed under the MIT License (see LICENSE for details)
Written by FracPete (fracpete at waikato dot ac dot nz)
"""

import os
import csv
import skimage
import skimage.io
import tensorflow as tf
from object import InferenceConfig
from mrcnn import model as modellib


def compress_mask(mask):
    """
    Compresses the mask, using simple run length encoding:
    X1:Y1;X2:Y2|...
    X:Y - single compression block
    Xn - the number of bits
    Yn - the mask, 0 or 1
    ; - separates compression blocks
    | - new line separator

    :param mask: the mask array
    :type mask: ndarray
    :return: the mask
    :rtype: str
    """

    return ""


def update_bboxes(model, image_path, bbox_path, label, out_path, verbose=0):
    """
    Updates the bounding boxes using the supplied model.

    :param model: the model to use for getting the object ROI
    :param image_path: the image to load and use
    :type image_path: str
    :param bbox_path: the CSV file with the bounding boxes
    :type bbox_path: str
    :param label: the label to restrict the update to
    :type label: str
    :param out_path: the CSV to store the updated bounding boxes in
    :type out_path: str
    :param verbose: the verbosity level, 0=off, higher number means more outout
    :type verbose: int
    """

    # read image
    image = skimage.io.imread(image_path)
    image = image[:, :, :3]

    # read csv
    bbox_rows = list()
    header = None
    req_cols = ["x0", "y0", "x1", "y1", "label_str", "score"]
    with open(bbox_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if header is None:
                header = row.keys()
                # check header
                for req_col in req_cols:
                    if req_col not in header:
                        raise Exception("Failed to locate column: " + req_col)
            bbox_rows.append(row)

    # created sorted header
    with open(bbox_path, 'r') as csvfile:
        header_sorted = csvfile.readline().strip().split(",")
        header_sorted.append("mask")

    # update bboxes
    bbox_updated = list()
    for row in bbox_rows:
        if row['label_str'] != label:
            bbox_updated.append(row)
            continue
        # crop image
        x0 = float(row['x0'])
        y0 = float(row['y0'])
        x1 = float(row['x1'])
        y1 = float(row['y1'])
        box = image[int(y0):int(y1), int(x0):int(x1)]
        # detect objects
        r = model.detect([box], verbose=verbose)[0]
        mask = ""
        if len(r['rois']) > 0:
            roi = r['rois'][0]
            y0N = int(y0) + roi[0]
            x0N = int(x0) + roi[1]
            y1N = int(y0) + roi[2]
            x1N = int(x0) + roi[3]
            if verbose > 1:
                print("original bbox coords (y0,x0,y1,x1):", y0, x0, y1, x1)
                print("new coords inside bbox (y0,x0,y1,x1):", roi)
                print("new bbox coords (y0,x0,y1,x1):", y0N, x0N, y1N, x1N)
            row['y0'] = str(y0N)
            row['x0'] = str(x0N)
            row['y1'] = str(y1N)
            row['x1'] = str(x1N)
            row['score'] = str(r['scores'][0])
            mask = compress_mask(r['masks'][0])
        row['mask'] = mask
        bbox_updated.append(row)

    # write output
    if verbose > 0:
        print("Saving updated bboxes to:", out_path)
    with open(out_path, 'w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=header_sorted)
        writer.writeheader()
        for row in bbox_updated:
            writer.writerow(row)


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect objects.')
    parser.add_argument('--image', required=True,
                        metavar="path to image",
                        help='The image with the objects to update the bounding boxes for.')
    parser.add_argument('--bbox', required=True,
                        metavar="path to CSV file",
                        help='CSV file with bounding information. Expected column names: x,y,w,h,label_str,score')
    parser.add_argument('--label', required=True,
                        metavar="the label to update",
                        help='The label to restrict the update to, eg "object"')
    parser.add_argument('--weights', required=True,
                        metavar="path to .h5 Keras file",
                        help="Path to Keras weights .h5 file")
    parser.add_argument('--out', required=True,
                        metavar="output CSV file",
                        help='The CSV file to store the updated bounding boxes in.')
    parser.add_argument('--gpu', required=False,
                        metavar="the GPU device ID to use",
                        help='On multi-GPU devices, limit the devices that tensorflow uses')
    parser.add_argument('--config', required=True,
                        metavar="path to YAML config file",
                        help='Configuration file for setting parameters')
    parser.add_argument('--verbose', required=False,
                        default=1,
                        metavar="verbosity level 0..N",
                        help='Verbosity level: 0=off, higher number means more output')
    args = parser.parse_args()

    print("image: ", args.image)
    print("bbox: ", args.bbox)
    print("label: ", args.label)
    print("weights: ", args.weights)
    print("out: ", args.out)
    print("Config : ", args.config)

    # GPU device
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    config = InferenceConfig()
    config.apply_yaml(args.config)
    config.display()

    # tensorflow tweaks
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.per_process_gpu_memory_fraction = config.tf_gpu_options_per_process_gpu_memory_fraction
    session = tf.InteractiveSession(config=tfconfig)

    # load model
    model = modellib.MaskRCNN(mode="inference", config=config,
                              model_dir=config.LOGS)
    model.load_weights(args.weights, by_name=True)

    # perform update
    update_bboxes(model, args.image, args.bbox, args.label, args.out)
