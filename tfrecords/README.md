# Converting ADAMS annotations

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
    ./venv/bin/pip install "tensorflow-gpu<2.0.0"
    ```
    
  * CPU only
  
    ```commandline
    ./venv/bin/pip install "tensorflow<2.0.0"
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

### Object Detection framework

Execute the `adams/object_detection.py` script with `-h` to see the help screen.

