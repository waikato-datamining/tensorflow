# MASK RCNN

* [Github](https://github.com/matterport/Mask_RCNN)
* [Blog post](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46)
* [Building Mask RCNN model](https://towardsdatascience.com/building-a-custom-mask-rcnn-model-with-tensorflow-object-detection-952f5b0c7ab4)
* [Pixel annotation tool](https://github.com/abreheret/PixelAnnotationTool)

## Pre-requisites

Installation based on: https://www.tensorflow.org/install/

* On Windows, install Build Tools 2015 for compilation support

  http://go.microsoft.com/fwlink/?LinkId=691126

* create virtualenv

  ```
  virtualenv -p /usr/bin/python3.5
  ```

* modules to install (using `<virtualenv>/bin/pip install`):

  * cython
  * numpy
  * tensorflow (tensorflow_gpu for GPU support)
  * jupyter
  * scikit-image
  * keras
  * imgaug
  * opencv-python
  * pycocotools (not on Windows)
  
* installing pycocotools on Windows via a modified repo

  ```
  <virtualenv>\bin\pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
  ```

## Installation

* install from repo
  
  ```
  <virtualenv>/bin/pip install git+https://github.com/matterport/Mask_RCNN.git
  ```

## Jupyter

* go in `MASK_RCNN` directory
* go into `samples`
* run `<virtualenv>/bin/jupyter`
* select `demo.ipynb`

**NB:** Use `%env CUDA_VISIBLE_DEVICES=2` to limit tensorflow to GPU with ID *2*, otherwise it will grab all the memory and thinks it is all available on a single GPU.

