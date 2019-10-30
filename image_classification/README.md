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

* For training, use the following script: `train.retrain`

### Predict

Once you have built a model, you can use as follows:

* For making predictions for a single image, use: `predict.label_image`
* For polling images in a directory and making continous predictions with CSV companion files, use: `predict.poll`
