# tensorflow
Customizations of tensorflow-based applications.

Currently available:
* [efficientdet](efficientdet) - Docker image for EfficientDet: Scalable and Efficient Object Detection
* [image-segmentation-keras](image-segmentation-keras) - Docker image for image segmentation using Keras
* [image classification](image_classification) - Docker images for training image classification models 
  and making predictions with them (uses TensorFlow 1)
* [image classification2](image_classification2) - Docker images for training image classification models 
  and making predictions with them (uses TensorFlow 2)
* [object detection](object_detection) - for building Docker images of TensorFlow's 
  object detection framework
* [tflite_model_maker](tflite_model_maker) - library for training models and making predictions using tflite's model maker
* [tfutils](tfutils) - library for some common TensorFlow operations

For generating tfrecords, please see the [wai.annotations](https://github.com/waikato-ufdl/wai-annotations) 
library, which can convert to and fro various file formats.
