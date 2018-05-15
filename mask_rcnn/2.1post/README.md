# Mask-RCNN object shape detection

This is based on the *balloon* example from the [Mask-RCNN](https://github.com/matterport/Mask_RCNN) project.

This is an example showing the use of Mask RCNN in a real application.
We train the model to detect objects only, and then we use the generated 
masks to keep objects in color while changing the rest of the image to
grayscale. 

## Installation
From the [Releases page](https://github.com/matterport/Mask_RCNN/releases) page:
1. Download `mask_rcnn_object.h5`. Save it in the root directory of the repo (the `mask_rcnn` directory).
2. Download `object_dataset.p3`. Expand it such that it's in the path `mask_rcnn/datasets/object/`.

## Apply color splash using the provided weights
Apply splash effect on an image:

```bash
python3 object.py splash --weights=/path/to/mask_rcnn/mask_rcnn_object.h5 --image=<file name or URL>
```

Apply splash effect on a video. Requires OpenCV 3.2+:

```bash
python3 object.py splash --weights=/path/to/mask_rcnn/mask_rcnn_object.h5 --video=<file name or URL>
```

## Train the Object model

Train a new model starting from pre-trained COCO weights
```
python3 object.py train --dataset=/path/to/object/dataset --weights=coco
```

Resume training a model that you had trained earlier
```
python3 object.py train --dataset=/path/to/object/dataset --weights=last
```

Train a new model starting from ImageNet weights
```
python3 object.py train --dataset=/path/to/object/dataset --weights=imagenet
```

The code in `object.py` is set to train for 3K steps (30 epochs of 100 steps each), and using a batch size of 2. 
Update the schedule to fit your needs.
