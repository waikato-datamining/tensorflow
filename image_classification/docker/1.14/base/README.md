# Labeling images

Allows labeling of images with Tensorflow's image classification capabilities, using Tensorflow 1.14.0.

## Docker

### Build local image

* Build the image from Docker file (from within /path_to/tensorflow/image_classification/docker/1.14/base)

  ```commandline
  docker build -t tfic_base .
  ```

* Run the container

  ```commandline
  docker run --runtime=nvidia -v /local:/container -it tfic_base
  ```

### Pre-built images

* Build

  ```commandline
  docker build -t tensorflow/image_classification:1.14 .
  ```
  
* Tag

  ```commandline
  docker tag \
    tensorflow/image_classification:1.14 \
    public-push.aml-repo.cms.waikato.ac.nz:443/tensorflow/image_classification:1.14
  ```
  
* Push

  ```commandline
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/tensorflow/image_classification:1.14
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```
  
* Pull

  If image is available in aml-repo and you just want to use it, you can pull using following command and then [run](#run).

  ```commandline
  docker pull public.aml-repo.cms.waikato.ac.nz:443/tensorflow/image_classification:1.14
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public.aml-repo.cms.waikato.ac.nz:443
  ```
  Then tag by running:
  
  ```commandline
  docker tag \
    public.aml-repo.cms.waikato.ac.nz:443/tensorflow/image_classification:1.14 \
    tensorflow/image_classification:1.14
  ```

* <a name="run">Run</a>

  ```commandline
  docker run --runtime=nvidia -v /local:/container -it tensorflow/image_classification:1.14
  ```
  "/local:/container" maps a local disk directory into a directory inside the container

