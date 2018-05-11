# MASK RCNN

* [Github](https://github.com/matterport/Mask_RCNN)
* [Blog post](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46)
* [Building Mask RCNN model](https://towardsdatascience.com/building-a-custom-mask-rcnn-model-with-tensorflow-object-detection-952f5b0c7ab4)
* [Pixel annotation tool](https://github.com/abreheret/PixelAnnotationTool)

## Pre-requisites

Installation based on: https://www.tensorflow.org/install/

* create virtualenv

  ```
  virtualenv -p /usr/bin/python3.5
  ```

* modules to install (using `<virtualenv>/bin/pip install`):

  * cython
  * numpy
  * tensorflow
  * jupyter
  * scikit-image
  * keras
  * imgaug
  * opencv-python
  * pycocotools

## Installation

* clone repo
  
  ```
  git clone https://github.com/matterport/Mask_RCNN.git
  ```

* change into `MASK_RCNN` directory
* run `<virtualenv>/bin/python setup.py install`

## Jupyter

* go in `MASK_RCNN` directory
* go into `samples`
* run `<virtualenv>/bin/jupyter`
* select `demo.ipynb`

