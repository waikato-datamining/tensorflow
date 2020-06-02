# EfficientDet

Mingxing Tan, Ruoming Pang, Quoc V. Le. EfficientDet: Scalable and Efficient Object Detection. CVPR 2020. 
Arxiv link: https://arxiv.org/abs/1911.09070

## Version

Code from here:

https://github.com/google/automl/tree/master/efficientdet

Using the following git hash:

```
786f9f459e52d9d90650b1635f200ffaf21c6677
```

## Notes

* requires sharded TFRecords

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
    -it public.aml-repo.cms.waikato.ac.nz:443/tensorflow/efficientdet:1.15.2_2020-05-24
  ```

  **NB:** For docker versions 19.03 (`docker version`) and newer, use `--gpus=all` instead of `--runtime=nvidia`.

* If need be, remove all containers and images from your system:

  ```commandline
  docker stop $(docker ps -a -q) && docker rm $(docker ps -a -q) && docker system prune -a
  ```


### Build local image

* Build image `eft` from Docker file (from within /path_to/tensorflow/efficientdet/1.15.2_2020-05-24/base)

  ```commandline
  docker build -t eft .
  ```
  
* Run image `eft` in interactive mode (i.e., using `bash`) as container `isk_container`

  ```commandline
  docker run --runtime=nvidia --name eft_container -ti -v \
    /path_to/local_disk/containing_data:/path_to/mount/inside/docker_container \
    eft bash
  ```

### Pre-built images

* Build

  ```commandline
  docker build -t tensorflow/eft:1.15.2_2020-05-24 .
  ```
  
* Tag

  ```commandline
  docker tag \
    tensorflow/eft:1.15.2_2020-05-24 \
    public-push.aml-repo.cms.waikato.ac.nz:443/tensorflow/efficientdet:1.15.2_2020-05-24
  ```
  
* Push

  ```commandline
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/tensorflow/efficientdet:1.15.2_2020-05-24
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```
  
* Pull

  If image is available in aml-repo and you just want to use it, you can pull using following command and then [run](#run).

  ```commandline
  docker pull public.aml-repo.cms.waikato.ac.nz:443/tensorflow/efficientdet:1.15.2_2020-05-24
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public.aml-repo.cms.waikato.ac.nz:443
  ```
  Then tag by running:
  
  ```commandline
  docker tag \
    public.aml-repo.cms.waikato.ac.nz:443/tensorflow/efficientdet:1.15.2_2020-05-24 \
    tensorflow/eft:1.15.2_2020-05-24
  ```
  
* <a name="run">Run</a>

  ```commandline
  docker run --runtime=nvidia -v /local:/container -it tensorflow/eft:1.15.2_2020-05-24
  ```
  `/local:/container` maps a local disk directory into a directory inside the container


### Command-line tools

The following command-line toosl are available:

* `ed_main` - calls `/opt/automl/efficientdet/main.py`
* `ed_model_inspect` - calls `/opt/automl/efficientdet/model_inspect.py`
* `ed_create_coco_tfrecord` - calls `/opt/automl/efficientdet/dataset/create_coco_tfrecord.py`
* `ed_create_pascal_tfrecord` - calls `/opt/automl/efficientdet/dataset/create_pascal_tfrecord.py`
