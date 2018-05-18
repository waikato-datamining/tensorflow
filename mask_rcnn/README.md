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

  * using virtualenv
  
    ```
    virtualenv -p /usr/bin/python3.5
    ```
    
  * using anaconda
  
    ```
    conda create --name tf-py35 python=3.5 numpy scipy h5py
    ```

* activate the environment

* install dependencies:

  ```
  pip install -r requirements.txt
  ```

* tensorflow

  * CPU only: `pip install tensorflow`
  * GPU: `pip install tensorflow_gpu`
  
* MS COCO tools

  * non-Windows: `pip install pycocotools`
  * Windows: `pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI`

## Installation

* install from repo
  
  ```
  pip install git+https://github.com/matterport/Mask_RCNN.git
  ```

## Jupyter

The following applies to the code from the original MASK_RCNN github repository:
* clone the directory using `git clone https://github.com/matterport/Mask_RCNN`
* go into `MASK_RCNN`
* go into `samples`
* run `jupyter`
* select `demo.ipynb`

**NB:** Use `%env CUDA_VISIBLE_DEVICES=2` to limit tensorflow to GPU with ID *2*, 
otherwise it will grab all the memory and thinks it is all available on a single GPU.

