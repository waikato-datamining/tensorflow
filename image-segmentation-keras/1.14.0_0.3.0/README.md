# Image Segmentation Keras

[Image Segmentation Keras : Implementation of Segnet, FCN, UNet, PSPNet and other models in Keras.](https://divamgupta.com/image-segmentation/2019/06/06/deep-learning-semantic-segmentation-keras.html), 
using Tensorflow 1.14.0. Using code from [here](https://github.com/divamgupta/image-segmentation-keras).

## Version

image-segmentation-keras version: 0.3.0

## Docker

### Quick start

* Log into registry using *public* credentials:

  ```commandline
  docker login -u public -p public public.aml-repo.cms.waikato.ac.nz:443 
  ```

* Pull and run image (adjust volume mappings `-v`):

  ```commandline
  docker run --runtime=nvidia \
    -v /local/dir:/container/dir \
    -it public.aml-repo.cms.waikato.ac.nz:443/tensorflow/image-segmentation-keras:1.14.0_0.3.0
  ```

  **NB:** For docker versions 19.03 (`docker version`) and newer, use `--gpus=all` instead of `--runtime=nvidia`.

* If need be, remove all containers and images from your system:

  ```commandline
  docker stop $(docker ps -a -q) && docker rm $(docker ps -a -q) && docker system prune -a
  ```


### Build local image

* Build image `isk` from Docker file (from within /path_to/tensorflow/image-segmentation-keras/1.14.0_0.3.0)

  ```commandline
  docker build -t isk .
  ```
  
* Run image `isk` in interactive mode (i.e., using `bash`) as container `isk_container`

  ```commandline
  docker run --runtime=nvidia --name isk_container -ti -v \
    /local/dir:/container/dir \
    isk bash
  ```

### Pre-built images

* Build

  ```commandline
  docker build -t tensorflow/isk:1.14.0_0.3.0 .
  ```
  
* Tag

  ```commandline
  docker tag \
    tensorflow/isk:1.14.0_0.3.0 \
    public-push.aml-repo.cms.waikato.ac.nz:443/tensorflow/image-segmentation-keras:1.14.0_0.3.0
  ```
  
* Push

  ```commandline
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/tensorflow/image-segmentation-keras:1.14.0_0.3.0
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```
  
* Pull

  If image is available in aml-repo and you just want to use it, you can pull using following command and then [run](#run).

  ```commandline
  docker pull public.aml-repo.cms.waikato.ac.nz:443/tensorflow/image-segmentation-keras:1.14.0_0.3.0
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public.aml-repo.cms.waikato.ac.nz:443
  ```
  Then tag by running:
  
  ```commandline
  docker tag \
    public.aml-repo.cms.waikato.ac.nz:443/tensorflow/image-segmentation-keras:1.14.0_0.3.0 \
    tensorflow/isk:1.14.0_0.3.0
  ```
  
* <a name="run">Run</a>

  ```commandline
  docker run --runtime=nvidia -v /local/dir:/container/dir -it tensorflow/isk:1.14.0_0.3.0
  ```
  `/local/dir:/container/dir` maps a local disk directory into a directory inside the container
