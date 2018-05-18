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


def mask_to_string(roi, mask):
    """
    Turns the mask into a string, row-wise with rows separated by ';'

    :param roi: the region of interest to restrict the mask to
    :type roi: ndarray (y0, x0, y1, x1)
    :param mask: the mask array
    :type mask: ndarray
    :return: the mask
    :rtype: str
    """

    all = list()
    for y in range(roi[0], roi[2]):
        line = list()
        for x in range(roi[1], roi[3]):
            line.append("1" if mask[y, x] else "0")
        all.append("".join(line))

    return ";".join(all)


def update_bboxes(model, image_path, bbox_path, label, out_path, verbose=0, mask=False):
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
    :param mask: whether to store the mask as well
    :type mask: bool
    """

    if verbose > 0:
        print("Processing image:", image_path)

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
        if mask:
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
        mask_str = ""
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
            if mask:
                mask_str = mask_to_string(roi, r['masks'])
        if mask:
            row['mask'] = mask_str
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
                        metavar="path to image or directory with images",
                        help='The image with the objects to update the bounding boxes for. '
                             + 'When supplying a directory, all PNG and JPG images that have a '
                             + 'corresponding .csv or -rois.csv file will get processed.')
    parser.add_argument('--bbox', required=False,
                        metavar="path to CSV file",
                        help='CSV file with bounding information. '
                             + 'Required column names (can have more): x0,y0,x1,y1,label_str,score. '
                             + 'Not required when "--image" points to directory.')
    parser.add_argument('--label', required=True,
                        metavar="the label to update",
                        help='The label to restrict the update to, eg "object"')
    parser.add_argument('--weights', required=True,
                        metavar="path to .h5 Keras file",
                        help="Path to Keras weights .h5 file")
    parser.add_argument('--out', required=False,
                        metavar="output CSV file",
                        help='The CSV file to store the updated bounding boxes in. '
                             + 'Not required when "--image" points to directory.')
    parser.add_argument('--mask', required=False,
                        default=False,
                        action='store_true',
                        help='Whether to store the mask as well (run length encoded format).')
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
    if os.path.isdir(args.image):
        for f in os.listdir(args.image):
            if not os.path.isfile(os.path.join(args.image, f)):
                continue
            if f.lower().endswith(".png") or f.lower().endswith(".jpg"):
                image = os.path.join(args.image, f)
                bbox1 = os.path.splitext(image)[0] + ".csv"
                bbox2 = os.path.splitext(image)[0] + "-rois.csv"
                out = os.path.splitext(image)[0] + "-updated.csv"
                if os.path.exists(bbox1):
                    update_bboxes(model, image, bbox1, args.label, out,
                                  verbose=args.verbose, mask=args.mask)
                elif os.path.exists(bbox2):
                    update_bboxes(model, image, bbox2, args.label, out,
                                  verbose=args.verbose, mask=args.mask)

    else:
        if args.bbox is None:
            raise Exception("No CSV file for bounding boxes provided!")
        if args.out is None:
            raise Exception("No output CSV file for updated bounding boxes provided!")
        update_bboxes(model, args.image, args.bbox, args.label, args.out,
                      verbose=args.verbose, mask=args.mask)
