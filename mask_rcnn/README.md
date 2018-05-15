# MASK RCNN

* [Github](https://github.com/matterport/Mask_RCNN)
* [Blog post](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46)
* [Building Mask RCNN model](https://towardsdatascience.com/building-a-custom-mask-rcnn-model-with-tensorflow-object-detection-952f5b0c7ab4)
* [VIA annotation tool](http://www.robots.ox.ac.uk/~vgg/software/via/)

## Pre-requisites

Installation based on: https://www.tensorflow.org/install/

* On Windows, install Build Tools 2015 for compilation support

  http://go.microsoft.com/fwlink/?LinkId=691126

* create virtualenv

  ```
  virtualenv -p /usr/bin/python3.5
  ```

* install dependencies:

  ```
  <virtualenv>/bin/pip install -r requirements.txt
  ```

* tensorflow

  * CPU only: `<virtualenv>/bin/pip install tensorflow`
  * GPU: `<virtualenv>/bin/pip install tensorflow_gpu`
  
* MS COCO tools

  * non-Windows: `<virtualenv>/bin/pip install pycocotools`
  * Windows: `<virtualenv>\bin\pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI`

## Installation

* install from repo
  
  ```
  <virtualenv>/bin/pip install git+https://github.com/matterport/Mask_RCNN.git
  ```

## Jupyter

The following applies to the code from the original MASK_RCNN github repository:
* go in `MASK_RCNN` directory
* go into `samples`
* run `<virtualenv>/bin/jupyter`
* select `demo.ipynb`

**NB:** Use `%env CUDA_VISIBLE_DEVICES=2` to limit tensorflow to GPU with ID *2*, otherwise it will grab all the memory and thinks it is all available on a single GPU.

