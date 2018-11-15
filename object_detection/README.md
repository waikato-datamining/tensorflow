# Object Detection framework

Allows processing of images with Tensorflow's Object Detection framework. 

## Installation

* install virtual environment

  ```commandline
  virtualenv -p /usr/bin/python2.7 venv
  ```

* install dependencies

  ```commandline
  ./venv/bin/pip install -r requirements.txt 
  ```

* install tensorflow

  * with GPU
  
    ```commandline
    ./venv/bin/pip install tensorflow-gpu
    ```
    
  * CPU only
  
    ```commandline
    ./venv/bin/pip install tensorflow
    ```
    
* install object detection framework ([instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md))

  ```commandline
  git clone https://github.com/tensorflow/models
  ```

* you need to add the directory above the `object_detection` one to the `PYTHONPATH`
  environment variable.
  
  ```commandline
  export PYTHONPATH=$PYTHONPATH:/some/where/models/research
  ```

## Usage

Execute the `predict.py` script with `-h` to see the help screen.

