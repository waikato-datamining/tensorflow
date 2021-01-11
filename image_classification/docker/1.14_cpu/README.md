# Labeling images

Allows labeling of images with Tensorflow's image classification capabilities, using Tensorflow 1.14.0 (CPU).

## Docker

### Quick start

* Log into registry using *public* credentials:

  ```commandline
  docker login -u public -p public public.aml-repo.cms.waikato.ac.nz:443 
  ```

* Pull and run image (adjust volume mappings `-v`):

  ```commandline
  docker run \
    -v /local/dir:/container/dir \
    -it public.aml-repo.cms.waikato.ac.nz:443/tensorflow/image_classification:1.14_cpu
  ```

* If need be, remove all containers and images from your system:

  ```commandline
  docker stop $(docker ps -a -q) && docker rm $(docker ps -a -q) && docker system prune -a
  ```


### Build local image

* Build the image from Docker file (from within /path_to/tensorflow/image_classification/docker/1.14_cpu)

  ```commandline
  docker build -t tfic_cpu .
  ```

* Run the container

  ```commandline
  docker run -v /local/dir:/container/dir -it tfic_cpu
  ```

### Pre-built images

* Build

  ```commandline
  docker build -t tensorflow/image_classification:1.14_cpu .
  ```
  
* Tag

  ```commandline
  docker tag \
    tensorflow/image_classification:1.14_cpu \
    public-push.aml-repo.cms.waikato.ac.nz:443/tensorflow/image_classification:1.14_cpu
  ```
  
* Push

  ```commandline
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/tensorflow/image_classification:1.14_cpu
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```
  
* Pull

  If image is available in aml-repo and you just want to use it, you can pull using following command and then [run](#run).

  ```commandline
  docker pull public.aml-repo.cms.waikato.ac.nz:443/tensorflow/image_classification:1.14_cpu
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public.aml-repo.cms.waikato.ac.nz:443
  ```
  Then tag by running:
  
  ```commandline
  docker tag \
    public.aml-repo.cms.waikato.ac.nz:443/tensorflow/image_classification:1.14_cpu \
    tensorflow/image_classification:1.14_cpu
  ```

* <a name="run">Run</a>

  ```commandline
  docker run -v /local/dir:/container/dir -it tensorflow/image_classification:1.14_cpu
  ```
  `/local/dir:/container/dir` maps a local disk directory into a directory inside the container


## Usage

The following command-line tools are available (see [here](../../README.md) for more details):

* `tfic-retrain` - for (re)training a pre-trained model on new data
* `tfic-stats` - for generating statistics for a trained model
* `tfic-labelimage` - labeling a single image
* `tfic-poll` - for batch or continuous predictions
  