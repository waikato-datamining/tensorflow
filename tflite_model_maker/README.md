# tflite model maker command-line utilities

Based on Jupyter Notebook located here:

https://github.com/tensorflow/tensorflow/blob/7d7cab61b0fe1bf8e01890fda9161c0f0c6e1a1a/tensorflow/lite/g3doc/tutorials/model_maker_object_detection.ipynb


## Installation

You can install the tools in a virtual environment as follows:

```commandline
pip install wai.tflite_model_maker
```

## Usage

### Training

You can use the `tmm-od-train` tool to train an object detection model:

```
usage: tmm-od-train [-h] --annotations FILE
                    [--model_type {efficientdet_lite0,efficientdet_lite1,efficientdet_lite2,efficientdet_lite3,efficientdet_lite4}]
                    [--hyper_params FILE] [--num_epochs INT]
                    [--batch_size INT] --output_dir DIR [--evaluate]

Trains an object detection model.

optional arguments:
  -h, --help            show this help message and exit
  --annotations FILE    The CSV file with the annotations. (default: None)
  --model_type {efficientdet_lite0,efficientdet_lite1,efficientdet_lite2,efficientdet_lite3,efficientdet_lite4}
                        The model architecture to use. (default:
                        efficientdet_lite0)
  --hyper_params FILE   The YAML file with hyper parameter settings. (default:
                        None)
  --num_epochs INT      The number of epochs to use for training (can also be
                        supplied through hyper parameters). (default: None)
  --batch_size INT      The batch size to use. (default: 8)
  --output_dir DIR      The directory to store the trained model in. (default:
                        None)
  --evaluate            If test data is part of the annotations, then the
                        resulting model can be evaluated against it. (default:
                        False)
```

### Prediction

For getting predictions for a single image, you can use the `tmm-od-predict` tool:

```
usage: tmm-od-predict [-h] --model FILE --labels FILE --image FILE
                      [--threshold 0-1] [--output FILE]

Uses an object detection model to make predictions on a single image.

optional arguments:
  -h, --help       show this help message and exit
  --model FILE     The tflite object detection model to use. (default: None)
  --labels FILE    The text file with the labels (one label per line).
                   (default: None)
  --image FILE     The image to make the prediction for. (default: None)
  --threshold 0-1  The probability threshold to use. (default: 0.3)
  --output FILE    The JSON file to store the predictions in. (default: None)
```

For using a file-polling approach for batch-processing images, you can use
the `tmm-od-predict-poll` tool:

```
usage: tmm-od-predict-poll [-h] --model FILE --labels FILE --prediction_in
                           PREDICTION_IN --prediction_out PREDICTION_OUT
                           [--prediction_tmp PREDICTION_TMP]
                           [--poll_wait POLL_WAIT] [--continuous]
                           [--use_watchdog]
                           [--watchdog_check_interval WATCHDOG_CHECK_INTERVAL]
                           [--delete_input] [--threshold 0-1] [--verbose]
                           [--quiet]

Uses an object detection model to make predictions on a single image.

optional arguments:
  -h, --help            show this help message and exit
  --model FILE          The tflite object detection model to use. (default:
                        None)
  --labels FILE         The text file with the labels (one label per line).
                        (default: None)
  --prediction_in PREDICTION_IN
                        Path to the test images (default: None)
  --prediction_out PREDICTION_OUT
                        Path to the output csv files folder (default: None)
  --prediction_tmp PREDICTION_TMP
                        Path to the temporary csv files folder (default: None)
  --poll_wait POLL_WAIT
                        poll interval in seconds when not using watchdog mode
                        (default: 1.0)
  --continuous          Whether to continuously load test images and perform
                        prediction (default: False)
  --use_watchdog        Whether to react to file creation events rather than
                        performing fixed-interval polling (default: False)
  --watchdog_check_interval WATCHDOG_CHECK_INTERVAL
                        check interval in seconds for the watchdog (default:
                        10.0)
  --delete_input        Whether to delete the input images rather than move
                        them to --prediction_out directory (default: False)
  --threshold 0-1       The probability threshold to use. (default: 0.3)
  --verbose             Whether to output more logging info (default: False)
  --quiet               Whether to suppress output (default: False)
```

For using a Redis backend, you can use the `tmm-od-predict-redis` tool:

```
usage: tmm-od-predict-redis [-h] [--redis_host HOST] [--redis_port PORT]
                            [--redis_db DB] --redis_in CHANNEL --redis_out
                            CHANNEL --model FILE --labels FILE
                            [--threshold 0-1] [--verbose]

Uses a tflite object detection model to make predictions on images received
via Redis and sends predictions back to Redis.

optional arguments:
  -h, --help           show this help message and exit
  --redis_host HOST    The redis server to connect to (default: localhost)
  --redis_port PORT    The port the redis server is listening on (default:
                       6379)
  --redis_db DB        The redis database to use (default: 0)
  --redis_in CHANNEL   The redis channel to receive the data from (default:
                       None)
  --redis_out CHANNEL  The redis channel to publish the processed data on
                       (default: None)
  --model FILE         The tflite object detection model to use. (default:
                       None)
  --labels FILE        The text file with the labels (one label per line).
                       (default: None)
  --threshold 0-1      The probability threshold to use. (default: 0.3)
  --verbose            Whether to output debugging information. (default:
                       False)
```