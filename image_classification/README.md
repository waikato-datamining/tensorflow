# Image classification with TensorFlow

Based on example code located here:

https://www.tensorflow.org/hub/tutorials/image_retraining


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
    ./venv/bin/pip install wai.tfimageclass
    ```

  * from source (from within the directory containing the `setup.py` script)::

    ```commandline
    ./venv/bin/pip install .
    ```

## Usage

All scripts support `--help` option to list all available options.

### Train

* For training, use module `wai.tfimageclass.train.retrain` or console script `tfic-retrain`
* For evaluating a built model, use module `wai.tfimageclass.train.stats` or console script `tfic-stats`

### Training data

All the data for building the model must be located in a single directory, with each sub-directory representing
a *label*. For instance for building a model for distinguishing flowers (daisy, dandelion, roses, sunflowers, tulip),
the data directory looks like this::

```
   |
   +- flowers
      |
      +- daisy
      |
      +- dandelion
      |
      +- roses
      |
      +- sunflowers
      |
      +- tulip
```


### Predict

Once you have built a model, you can use as follows:

* For making predictions for a single image, use module `wai.tfimageclass.predict.label_image` or console 
  script `tfic-labelimage`
* For polling images in a directory and making continous predictions with CSV companion files, use 
  module `wai.tfimageclass.predict.poll` or console script `tfic-poll`
