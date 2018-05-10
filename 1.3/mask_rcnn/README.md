# MASK RCNN

* [Github](https://github.com/matterport/Mask_RCNN)
* [Blog post](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46)

## Pre-requisites

* create virtualenv

  ```
  virtualenv -p /usr/bin/python3.5
  ```

* modules to install (using `<virtualenv>/bin/pip install`):

  * tensorflow
  * jupyter
  * scikit-image
  * keras
  * imgaug
  * opencv-python
  * pycocotools
  * cython

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

