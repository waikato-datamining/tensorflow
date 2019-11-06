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

Execute `tfrecords-convert -h` to see the help screen.

## References

### Tensorflow

This package uses a copy of the object detection framework from [tensorflow](https://github.com/tensorflow/models). The
copy was made on the 6th of November, 2019, from commit b9ef963d1e84da0bb9c0a6039457c39a699ea149. For the original
licence, see ``src/wai/tfrecords/object_detection/LICENSE``. For the procedure used to include the package, and the
modifications made to it, see ``src/wai/tfrecords/object_detection/__init.py__``.
