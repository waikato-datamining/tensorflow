The *wai.tflite_model_maker* library offers command-line tools for training tflite
models and applying them.

The following domains are supported:

* image classification
* object detection

For making predictions, the following approaches are available:

* single image
* batch/continuous processing using file-polling
* continuous processing via a Redis backend (for receiving images and sending predictions back to)
