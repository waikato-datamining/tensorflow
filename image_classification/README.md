# Image classification with TensorFlow

Based on example located here:

https://www.tensorflow.org/hub/tutorials/image_retraining


## Installation

* install virtual environment

  ```commandline
  virtualenv -p /usr/bin/python3.7 venv
  ```

* install dependencies

  ```commandline
  ./venv/bin/pip install -r requirements.txt 
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

## Usage

All scripts support `--help` option to list all available options.

### Train

* For training, use module `wai.tfimageclass.train.retrain` or console script `tfic-retrain`
* For evaluating a built model, use module `wai.tfimageclass.train.stats` or console script `tfic-stats`

### Predict

Once you have built a model, you can use as follows:

* For making predictions for a single image, use module `wai.tfimageclass.predict.label_image` or console 
  script `tfic-labelimage`
* For polling images in a directory and making continous predictions with CSV companion files, use 
  module `wai.tfimageclass.predict.poll` or console script `tfic-poll`
