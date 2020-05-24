# image-segmentation-keras

[Image Segmentation Keras : Implementation of Segnet, FCN, UNet, PSPNet and other models in Keras.](https://divamgupta.com/image-segmentation/2019/06/06/deep-learning-semantic-segmentation-keras.html), 
using Tensorflow 1.14.0. Using code from [here](https://github.com/divamgupta/image-segmentation-keras).

## Version

image-segmentation-keras version: 0.3.0

## Docker

### Build local image

* Build image `isk` from Docker file (from within /path_to/tensorflow/image-segmentation-keras/1.14.0_0.3.0/base)

  ```commandline
  docker build -t isk .
  ```
  
* Run image `isk` in interactive mode (i.e., using `bash`) as container `isk_container`

  ```commandline
  docker run --runtime=nvidia --name isk_container -ti -v \
    /path_to/local_disk/containing_data:/path_to/mount/inside/docker_container \
    isk bash
  ```

### Pre-built images

* Build

  ```commandline
  docker build -t tensorflow/isk:1.14.0_2019-02-22 .
  ```
  
* Tag

  ```commandline
  docker tag \
    tensorflow/isk:1.14.0_2019-02-22 \
    public-push.aml-repo.cms.waikato.ac.nz:443/tensorflow/image-segmentation-keras:1.14.0_2019-02-22
  ```
  
* Push

  ```commandline
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/tensorflow/image-segmentation-keras:1.14.0_2019-02-22
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```
  
* Pull

  If image is available in aml-repo and you just want to use it, you can pull using following command and then [run](#run).

  ```commandline
  docker pull public.aml-repo.cms.waikato.ac.nz:443/tensorflow/image-segmentation-keras:1.14.0_2019-02-22
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public.aml-repo.cms.waikato.ac.nz:443
  ```
  Then tag by running:
  
  ```commandline
  docker tag \
    public.aml-repo.cms.waikato.ac.nz:443/tensorflow/image-segmentation-keras:1.14.0_2019-02-22 \
    tensorflow/isk:1.14.0_2019-02-22
  ```
  
* <a name="run">Run</a>

  ```commandline
  docker run --runtime=nvidia -v /local:/container -it tensorflow/isk:1.14.0_2019-02-22
  ```
  `/local:/container` maps a local disk directory into a directory inside the container
