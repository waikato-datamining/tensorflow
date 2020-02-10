# Inference with Object Detection framework

Allows processing of images with Tensorflow's Object Detection framework, using Tensorflow 2.1.0.

## Version

Object Detection framework github repo hash:

```
d7e4cc8a3ef9b7c20d7212649dade80989a8c324
```

and timestamp:

```
31 Jan 2020
```

## Docker

## Build local image

* Build the image from Docker file (from within /path_to/tensorflow/object_detection/2.1.0_2020-01-31/predict)

  ```commandline
  docker build -t tf_predict .
  ```
  
* Run the container

  ```commandline
  docker run --runtime=nvidia --name tf_container -ti -v \
    /path_to/local_disk/containing_data:/path_to/mount/inside/docker_container tf_predict \
    --graph /path_to/your_data/output/exported_graphs/frozen_inference_graph.pb \
    --labels /path_to/your_data_label_map.pbtxt --prediction_in /path_to/your_data/test_images/ \
    --prediction_out /path_to/your_data/output/results --score 0.1 --num_imgs 3 --num_classes 1
  ```

## Pre-built images

* Build

  ```commandline
  docker build -t tensorflow/object_detection:2.1.0_2020-01-31_predict .
  ```
  
* Tag

  ```commandline
  docker tag \
    tensorflow/object_detection:2.1.0_2020-01-31_predict \
    public-push.aml-repo.cms.waikato.ac.nz:443/tensorflow/object_detection:2.1.0_2020-01-31_predict
  ```
  
* Push

  ```commandline
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/tensorflow/object_detection:2.1.0_2020-01-31_predict
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```
  
* Pull

  If image is available in aml-repo and you just want to use it, you can pull using following command and then [run](#run).

  ```commandline
  docker pull public.aml-repo.cms.waikato.ac.nz:443/tensorflow/object_detection:2.1.0_2020-01-31_predict
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public.aml-repo.cms.waikato.ac.nz:443
  ```
  Then tag by running:
  
  ```commandline
  docker tag \
    public.aml-repo.cms.waikato.ac.nz:443/tensorflow/object_detection:2.1.0_2020-01-31_predict \
    tensorflow/object_detection:2.1.0_2020-01-31_predict
  ```

* <a name="run">Run</a>

  ```commandline
  docker run --runtime=nvidia -v /local:/container -it tensorflow/object_detection:2.1.0_2020-01-31_predict \
    --graph /path_to/your_data/output/exported_graphs/frozen_inference_graph.pb \
    --labels /path_to/your_data_label_map.pbtxt --prediction_in /path_to/your_data/test_images/ \
    --prediction_out /path_to/your_data/output/results --score 0.1 --num_imgs 3 --num_classes 1
  ```
  "/local:/container" maps a local disk directory into a directory inside the container

