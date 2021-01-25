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

  ```
  usage: tfic-retrain [-h] [--image_dir IMAGE_DIR]
                      [--image_lists_dir IMAGE_LISTS_DIR]
                      [--output_graph OUTPUT_GRAPH] [--output_info OUTPUT_INFO]
                      [--intermediate_output_graphs_dir INTERMEDIATE_OUTPUT_GRAPHS_DIR]
                      [--intermediate_store_frequency INTERMEDIATE_STORE_FREQUENCY]
                      [--output_labels OUTPUT_LABELS]
                      [--summaries_dir SUMMARIES_DIR]
                      [--training_steps TRAINING_STEPS]
                      [--learning_rate LEARNING_RATE]
                      [--testing_percentage TESTING_PERCENTAGE]
                      [--validation_percentage VALIDATION_PERCENTAGE]
                      [--eval_step_interval EVAL_STEP_INTERVAL]
                      [--train_batch_size TRAIN_BATCH_SIZE]
                      [--test_batch_size TEST_BATCH_SIZE]
                      [--validation_batch_size VALIDATION_BATCH_SIZE]
                      [--print_misclassified_test_images]
                      [--bottleneck_dir BOTTLENECK_DIR]
                      [--final_tensor_name FINAL_TENSOR_NAME]
                      [--flip_left_right] [--random_crop RANDOM_CROP]
                      [--random_scale RANDOM_SCALE]
                      [--random_brightness RANDOM_BRIGHTNESS]
                      [--tfhub_module TFHUB_MODULE]
                      [--saved_model_dir SAVED_MODEL_DIR]
                      [--logging_verbosity {DEBUG,INFO,WARN,ERROR,FATAL}]
                      [--checkpoint_path CHECKPOINT_PATH]

  Trains a pre-trained model with new images.

  optional arguments:
    -h, --help            show this help message and exit
    --image_dir IMAGE_DIR
                          Path to folders of labeled images. (default: )
    --image_lists_dir IMAGE_LISTS_DIR
                          Where to save the lists of images used for training,
                          validation and testing (in JSON); ignored if directory
                          does not exist. (default: None)
    --output_graph OUTPUT_GRAPH
                          Where to save the trained graph. (default:
                          /tmp/output_graph.pb)
    --output_info OUTPUT_INFO
                          Whether to save the (optional) information about the
                          graph, like image dimensions and layers, (in JSON);
                          ignored if not supplied. (default: None)
    --intermediate_output_graphs_dir INTERMEDIATE_OUTPUT_GRAPHS_DIR
                          Where to save the intermediate graphs. (default:
                          /tmp/intermediate_graph/)
    --intermediate_store_frequency INTERMEDIATE_STORE_FREQUENCY
                          How many steps to store intermediate graph. If "0"
                          then will not store. (default: 0)
    --output_labels OUTPUT_LABELS
                          Where to save the trained graph's labels. (default:
                          /tmp/output_labels.txt)
    --summaries_dir SUMMARIES_DIR
                          Where to save summary logs for TensorBoard. (default:
                          /tmp/retrain_logs)
    --training_steps TRAINING_STEPS
                          How many training steps to run before ending.
                          (default: 4000)
    --learning_rate LEARNING_RATE
                          How large a learning rate to use when training.
                          (default: 0.01)
    --testing_percentage TESTING_PERCENTAGE
                          What percentage of images to use as a test set.
                          (default: 10)
    --validation_percentage VALIDATION_PERCENTAGE
                          What percentage of images to use as a validation set.
                          (default: 10)
    --eval_step_interval EVAL_STEP_INTERVAL
                          How often to evaluate the training results. (default:
                          10)
    --train_batch_size TRAIN_BATCH_SIZE
                          How many images to train on at a time. (default: 100)
    --test_batch_size TEST_BATCH_SIZE
                          How many images to test on. This test set is only used
                          once, to evaluate the final accuracy of the model
                          after training completes. A value of -1 causes the
                          entire test set to be used, which leads to more stable
                          results across runs. (default: -1)
    --validation_batch_size VALIDATION_BATCH_SIZE
                          How many images to use in an evaluation batch. This
                          validation set is used much more often than the test
                          set, and is an early indicator of how accurate the
                          model is during training. A value of -1 causes the
                          entire validation set to be used, which leads to more
                          stable results across training iterations, but may be
                          slower on large training sets. (default: 100)
    --print_misclassified_test_images
                          Whether to print out a list of all misclassified test
                          images. (default: False)
    --bottleneck_dir BOTTLENECK_DIR
                          Path to cache bottleneck layer values as files.
                          (default: /tmp/bottleneck)
    --final_tensor_name FINAL_TENSOR_NAME
                          The name of the output classification layer in the
                          retrained graph. (default: final_result)
    --flip_left_right     Whether to randomly flip half of the training images
                          horizontally. (default: False)
    --random_crop RANDOM_CROP
                          A percentage determining how much of a margin to
                          randomly crop off the training images. (default: 0)
    --random_scale RANDOM_SCALE
                          A percentage determining how much to randomly scale up
                          the size of the training images by. (default: 0)
    --random_brightness RANDOM_BRIGHTNESS
                          A percentage determining how much to randomly multiply
                          the training image input pixels up or down by.
                          (default: 0)
    --tfhub_module TFHUB_MODULE
                          Which TensorFlow Hub module to use. For more options,
                          search https://tfhub.dev for image feature vector
                          modules. (default: https://tfhub.dev/google/imagenet/i
                          nception_v3/feature_vector/3)
    --saved_model_dir SAVED_MODEL_DIR
                          Where to save the exported graph. (default: )
    --logging_verbosity {DEBUG,INFO,WARN,ERROR,FATAL}
                          How much logging output should be produced. (default:
                          INFO)
    --checkpoint_path CHECKPOINT_PATH
                          Where to save checkpoint files. (default:
                          /tmp/_retrain_checkpoint)
  ```

* For evaluating a built model, use module `wai.tfimageclass.train.stats` or console script `tfic-stats`

  ```
  usage: tfic-stats [-h] [--image_dir IMAGE_DIR] [--image_list IMAGE_LIST]
                    --graph GRAPH [--info INFO] [--labels LABELS]
                    [--input_height INPUT_HEIGHT] [--input_width INPUT_WIDTH]
                    [--input_layer INPUT_LAYER] [--output_layer OUTPUT_LAYER]
                    [--input_mean INPUT_MEAN] [--input_std INPUT_STD]
                    --output_preds OUTPUT_PREDS --output_stats OUTPUT_STATS
                    [--logging_verbosity {DEBUG,INFO,WARN,ERROR,FATAL}]

  Generates statistics in CSV format by recording predictions on images list
  files.

  optional arguments:
    -h, --help            show this help message and exit
    --image_dir IMAGE_DIR
                          Path to folders of labeled images. (default: )
    --image_list IMAGE_LIST
                          The JSON file with images per sub-directory. (default:
                          None)
    --graph GRAPH         graph/model to be executed (default: None)
    --info INFO           name of json file with model info (dimensions,
                          layers); overrides input_height/input_width/labels/inp
                          ut_layer/output_layer options (default: None)
    --labels LABELS       name of file containing labels (default: None)
    --input_height INPUT_HEIGHT
                          input height (default: 299)
    --input_width INPUT_WIDTH
                          input width (default: 299)
    --input_layer INPUT_LAYER
                          name of input layer (default: Placeholder)
    --output_layer OUTPUT_LAYER
                          name of output layer (default: final_result)
    --input_mean INPUT_MEAN
                          input mean (default: 0)
    --input_std INPUT_STD
                          input std (default: 255)
    --output_preds OUTPUT_PREDS
                          The CSV file to store the predictions in. (default:
                          None)
    --output_stats OUTPUT_STATS
                          The CSV file to store the statistics in. (default:
                          None)
    --logging_verbosity {DEBUG,INFO,WARN,ERROR,FATAL}
                          How much logging output should be produced. (default:
                          INFO)
  ```

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


### Export

Once you have built a model, you can export it to [Tensorflow lite](https://www.tensorflow.org/lite/),
using the `wai.tfimageclass.train.export` module or the `tfic-export` command-line tool:

```commandline
usage: tfic-export [-h] --saved_model_dir SAVED_MODEL_DIR --tflite_model
                   TFLITE_MODEL

Exports a Tensorflow model as Tensorflow lite one.

optional arguments:
  -h, --help            show this help message and exit
  --saved_model_dir SAVED_MODEL_DIR
                        Path to the saved Tensorflow model directory.
                        (default: )
  --tflite_model TFLITE_MODEL
                        The file to export the Tensorflow lite model to.
                        (default: )
```


### Predict

Once you have built a model (Tensorflow or Tensorflow lite), you can use it as follows:

* For making predictions for a single image, use module `wai.tfimageclass.predict.label_image` or console 
  script `tfic-labelimage`
  
  ```
  usage: tfic-labelimage [-h] --image IMAGE --graph GRAPH [--graph_type TYPE]
                         [--info INFO] [--labels LABELS]
                         [--input_height INPUT_HEIGHT]
                         [--input_width INPUT_WIDTH] [--input_layer INPUT_LAYER]
                         [--output_layer OUTPUT_LAYER] [--input_mean INPUT_MEAN]
                         [--input_std INPUT_STD] [--top_x TOP_X]
                         [--output_format TYPE] [--output_file FILE]

  Outputs predictions for single image using a trained model.

  optional arguments:
    -h, --help            show this help message and exit
    --image IMAGE         image to be processed (default: None)
    --graph GRAPH         graph/model to be executed (default: None)
    --graph_type TYPE     the type of graph/model to be loaded (default:
                          tensorflow)
    --info INFO           name of json file with model info (dimensions,
                          layers); overrides input_height/input_width/labels/inp
                          ut_layer/output_layer options (default: None)
    --labels LABELS       name of file containing labels (default: None)
    --input_height INPUT_HEIGHT
                          input height (default: 299)
    --input_width INPUT_WIDTH
                          input width (default: 299)
    --input_layer INPUT_LAYER
                          name of input layer (default: Placeholder)
    --output_layer OUTPUT_LAYER
                          name of output layer (default: final_result)
    --input_mean INPUT_MEAN
                          input mean (default: 0)
    --input_std INPUT_STD
                          input std (default: 255)
    --top_x TOP_X         output only the top K labels; use <1 for all (default:
                          5)
    --output_format TYPE  the output format for the predictions (default:
                          plaintext)
    --output_file FILE    the file to write the predictions, uses stdout if not
                          provided (default: None)
  ```
  
* For polling images in a directory and making continous predictions with CSV companion files, use 
  module `wai.tfimageclass.predict.poll` or console script `tfic-poll`
  
  ```
  usage: tfic-poll [-h] --in_dir DIR --out_dir DIR [--continuous] [--delete]
                   --graph FILE [--graph_type TYPE] [--info INFO]
                   [--labels FILE] [--input_height INT] [--input_width INT]
                   [--input_layer NAME] [--output_layer NAME] [--input_mean INT]
                   [--input_std INT] [--top_x INT] [--reset_session INT]
                   [--output_format TYPE]

  For bulk or continuous prediction output using a trained model.

  optional arguments:
    -h, --help            show this help message and exit
    --in_dir DIR          the input directory to poll for images (default: None)
    --out_dir DIR         the output directory for processed images and
                          predictions (default: None)
    --continuous          Whether to continuously load test images and perform
                          prediction (default: False)
    --delete              Whether to delete images rather than move them to the
                          output directory. (default: False)
    --graph FILE          graph/model to be executed (default: None)
    --graph_type TYPE     the type of graph/model to be loaded (default:
                          tensorflow)
    --info INFO           name of json file with model info (dimensions,
                          layers); overrides input_height/input_width/labels/inp
                          ut_layer/output_layer options (default: None)
    --labels FILE         name of file containing labels (default: None)
    --input_height INT    input height (default: 299)
    --input_width INT     input width (default: 299)
    --input_layer NAME    name of input layer (default: Placeholder)
    --output_layer NAME   name of output layer (default: final_result)
    --input_mean INT      input mean (default: 0)
    --input_std INT       input std (default: 255)
    --top_x INT           output only the top K labels; use <1 for all (default:
                          5)
    --reset_session INT   The number of processed images after which to
                          reinitialize the Tensorflow session to reduce memory
                          leaks. (default: 50)
    --output_format TYPE  the output format for the predictions (default: csv)
  ```
