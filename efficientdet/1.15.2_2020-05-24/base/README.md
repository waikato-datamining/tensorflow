# EfficientDet

Mingxing Tan, Ruoming Pang, Quoc V. Le. EfficientDet: Scalable and Efficient Object Detection. CVPR 2020. 
Arxiv link: https://arxiv.org/abs/1911.09070

## Version

git hash

```
786f9f459e52d9d90650b1635f200ffaf21c6677
```

## Docker

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
