# TensorFlow Utils

Library with utility functions for TensforFlow, to cut down on the amount 
of code to write for making predictions.


## Installation

* install virtual environment

  ```commandline
  virtualenv -p /usr/bin/python3.7 venv
  ```

* install tensorflow (1.x or 2.x works)

  * with GPU (1.x)
  
    ```commandline
    ./venv/bin/pip install "tensorflow-gpu<2.0.0"
    ```

  * with GPU (2.x)
  
    ```commandline
    ./venv/bin/pip install "tensorflow-gpu>=2.0.0"
    ```
    
  * CPU only (1.x)
  
    ```commandline
    ./venv/bin/pip install "tensorflow<2.0.0"
    ```
    
  * CPU only (2.x)
  
    ```commandline
    ./venv/bin/pip install "tensorflow>=2.0.0"
    ```

* install library

  * via pip

    ```commandline
    ./venv/bin/pip install wai.tfutils
    ```

  * from source (from within the directory containing the `setup.py` script)::

    ```commandline
    ./venv/bin/pip install .
    ```

## Library

* `wai.tfutils.load_frozen_graph` - for loading a frozen graph from disk
* `wai.tfutils.load_labels` - for loading the label map and categories from a protobuf text
* `wai.tfutils.inference_for_image` - for generating predictions for a single image
