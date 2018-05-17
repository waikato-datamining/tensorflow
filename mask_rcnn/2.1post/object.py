"""
Mask R-CNN
Train on the toy Object dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Copyright (C) 2018 University of Waikato, Hamilton, NZ
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 object.py train --dataset=/path/to/object/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 object.py train --dataset=/path/to/object/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 object.py train --dataset=/path/to/object/dataset --weights=imagenet

    # Apply color splash to an image
    python3 object.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 object.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import numpy as np
import skimage.draw
import tensorflow as tf
import yaml
import cv2
from mrcnn.config import Config
from mrcnn import model as modellib, utils


############################################################
#  Configurations
############################################################


class ObjectConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + object

    # Number of epochs to perform
    NUM_EPOCHS = 30

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # the model to use
    COCO_WEIGHTS_PATH = None

    # the log directory
    LOGS = "./logs"

    # avoid tensorflow grabbing too much memory
    tf_gpu_options_per_process_gpu_memory_fraction = 0.4

    def reinitialize(self):
        """
        Re-initializes some parameters.
        """
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM, 3])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, 3])

        # Image meta data length
        # See compose_image_meta() for details
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES

    def apply_yaml(self, conf_file):
        """
        Applies the parameters from the YAML file.

        :param conf_file: the yaml file to load and apply
        :type conf_file: str
        """
        with open(conf_file, 'r') as f:
            conf = yaml.load(f, Loader=yaml.Loader)
            for k in conf:
                if hasattr(self, k):
                    curr = getattr(self, k)
                    # force tuple
                    if isinstance(curr, tuple):
                        setattr(self, k, tuple(conf[k]))
                    # force ndarray
                    elif isinstance(curr, np.ndarray):
                        setattr(self, k, np.asarray(conf[k]))
                    # as is
                    else:
                        setattr(self, k, conf[k])
                else:
                    raise Exception("Unsupported parameter: " + k)
        self.reinitialize()

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:50} {}".format(a, getattr(self, a)))
        print("\n")


class InferenceConfig(ObjectConfig):
    """
    Set batch size to 1 since we'll be running inference on
    one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    """

    def reinitialize(self):
        self.GPU_COUNT = 1
        self.IMAGES_PER_GPU = 1
        super().reinitialize()


############################################################
#  Dataset
############################################################

class ViaObjectDataset(utils.Dataset):

    def load_object(self, dataset_dir, subset):
        """Load a subset of the Object dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("object", 1, "object")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions'].values()]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "object",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a object dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model, dataset):
    """
    Train the model.

    :param model: the model to load and train
    :param dataset: the base directory of the dataset, above "train" and "val"
    :type dataset: str
    """

    # Training dataset.
    dataset_train = ViaObjectDataset()
    dataset_train.load_object(dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ViaObjectDataset()
    dataset_val.load_object(dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=config.NUM_EPOCHS,
                layers='heads')


def color_splash(image, mask):
    """
    Apply color splash effect.

    :param image: RGB image [height, width, 3]
    :param mask: instance segmentation mask [height, width, instance count]
    :return: result image.
    """

    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None, output_dir=None):
    """
    Detects the objects and highlights them in the generated output.

    :param model: the model to use
    :param image_path: the image or image directory
    :type image_path: str
    :param video_path: the video or video directory
    :type video_path: str
    :param output_dir: the directory for storing the generated output
    :type output_dir: str
    """

    assert image_path or video_path

    if output_dir is None:
        output_dir = "."

    # Image or video?
    if image_path:
        # directory?
        if os.path.isdir(image_path):
            for f in os.listdir(image_path):
                if f.lower().endswith(".png") or f.lower().endswith(".jpg"):
                    detect_and_color_splash(model, image_path=os.path.join(image_path, f),
                                            output_dir=output_dir)
            return

        # Run model detection and generate the color splash effect
        print("Running on {}".format(image_path))
        # Read image
        image = skimage.io.imread(image_path)
        # make sure that there is no alpha channel
        image = image[:, :, :3]
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = os.path.join(output_dir, os.path.basename(image_path))
        skimage.io.imsave(file_name, splash)

    elif video_path:
        # directory?
        if os.path.isdir(video_path):
            for f in os.listdir(video_path):
                if f.lower().endswith(".mp4") or f.lower().endswith(".avi"):
                    detect_and_color_splash(model, video_path=os.path.join(video_path, f),
                                            output_dir=output_dir)
            return

        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = os.path.join(output_dir, os.path.basename(video_path))
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # make sure that there is no alpha channel
                image = image[:, :, :3]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vcapture.release()
        vwriter.release()

    print("Saved to ", file_name)


def bbox_to_csv(id, rois, scores, csv_file, append=False):
    """
    Stores the bounding box in the specified CSV file.

    :param id: the ID to use, eg image name or timeframe
    :type id: str
    :param rois: the bounding boxes (regions of interest aka rois)
    :type rois: array of ndarray
    :param scores: the scores (probabilities)
    :type scores: array of ndarray
    :param csv_file: the CSV file to store the the bounding box in
    :type csv_file: str
    :param append: if False, a head will get output, otherwise the data just gets appended
    :type append: bool
    """
    if not append:
        flags = 'w'
    else:
        flags = 'a'
    with open(csv_file, flags) as f:
        if not append:
            f.write("id,index,x0,y0,x1,y1,score\n")
        for idx, roi in enumerate(rois):
            f.write("%s,%i,%f,%f,%f,%f,%f\n" % (id, idx, roi[1], roi[0], roi[3], roi[2], scores[idx]))


def detect_and_bbox(model, image_path=None, video_path=None, output_dir=None):
    """
    Detects the objects and stores the bounding boxes in CSV file(s).

    :param model: the model to use
    :param image_path: the image or image directory
    :type image_path: str
    :param video_path: the video or video directory
    :type video_path: str
    :param output_dir: the directory for storing the generated output
    :type output_dir: str
    """

    assert image_path or video_path

    if output_dir is None:
        output_dir = "."

    # Image or video?
    if image_path:
        # directory?
        if os.path.isdir(image_path):
            for f in os.listdir(image_path):
                if f.lower().endswith(".png") or f.lower().endswith(".jpg"):
                    detect_and_bbox(model, image_path=os.path.join(image_path, f),
                                            output_dir=output_dir)
            return

        # Run model detection and generate the color splash effect
        print("Running on {}".format(image_path))
        # Read image
        image = skimage.io.imread(image_path)
        # make sure that there is no alpha channel
        image = image[:, :, :3]
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # bbox output
        file_name = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + ".csv")
        bbox_to_csv(image_path, r['rois'], r['scores'], file_name)

    elif video_path:
        # directory?
        if os.path.isdir(video_path):
            for f in os.listdir(video_path):
                if f.lower().endswith(".mp4") or f.lower().endswith(".avi"):
                    detect_and_bbox(model, video_path=os.path.join(video_path, f),
                                            output_dir=output_dir)
            return

        # Video capture
        vcapture = cv2.VideoCapture(video_path)

        # Define codec and create video writer
        file_name = os.path.join(output_dir, os.path.splitext(os.path.basename(video_path))[0] + ".csv")
        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # make sure that there is no alpha channel
                image = image[:, :, :3]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # bbox
                bbox_to_csv(video_path + "|" + count, r['rois'], r['scores'], file_name, append=(count > 0))
                count += 1
        vcapture.release()

    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect objects.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train', 'splash' or 'bbox'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/object/dataset/",
                        help='Directory of the Object dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image or image directory",
                        help='Image to apply the color splash effect on; if directory provided applies it to PNG and JPG images')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on; if directory provided applies it to MP4 or AVI videos')
    parser.add_argument('--output_dir', required=False,
                        metavar="output directory",
                        help='The directory to use for storing the output from the "splash" command')
    parser.add_argument('--gpu', required=False,
                        metavar="the GPU device ID to use",
                        help='On multi-GPU devices, limit the devices that tensorflow uses')
    parser.add_argument('--config', required=True,
                        metavar="path to YAML config file",
                        help='Configuration file for setting parameters')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"
    elif args.command == "bbox":
        assert args.image or args.video,\
               "Provide --image or --video to generate bounding box output"

    print("Command: ", args.command)
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Config : ", args.config)

    # GPU device
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Configurations
    if args.command == "train":
        config = ObjectConfig()
    else:
        config = InferenceConfig()
    config.apply_yaml(args.config)
    config.display()

    # tensorflow tweaks
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.per_process_gpu_memory_fraction = config.tf_gpu_options_per_process_gpu_memory_fraction
    session = tf.InteractiveSession(config=tfconfig)

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=config.LOGS)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=config.LOGS)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = config.COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video,
                                output_dir=args.output_dir)
    elif args.command == "bbox":
        detect_and_bbox(model, image_path=args.image,
                                video_path=args.video,
                                output_dir=args.output_dir)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
