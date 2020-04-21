# UNet

[U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/), 
using Tensorflow 1.14.0. Using code from [here](https://github.com/zhixuhao/unet).

## Version

unet github repo hash:

```
b45af4d458437d8281cc218a07fd4380818ece4a
```

and timestamp:

```
Feb 22, 2019
```

## Docker

### Build local image

* Build image `unet` from Docker file (from within /path_to/tensorflow/unet/1.14.0_2010-02-22/base)

  ```commandline
  docker build -t unet .
  ```
  
* Run image `unet` in interactive mode (i.e., using `bash`) as container `unet_container`

  ```commandline
  docker run --runtime=nvidia --name unet_container -ti -v \
    /path_to/local_disk/containing_data:/path_to/mount/inside/docker_container \
    unet bash
  ```

### Pre-built images

* Build

  ```commandline
  docker build -t tensorflow/unet:1.14.0_2010-02-22 .
  ```
  
* Tag

  ```commandline
  docker tag \
    tensorflow/unet:1.14.0_2010-02-22 \
    public-push.aml-repo.cms.waikato.ac.nz:443/tensorflow/unet:1.14.0_2010-02-22
  ```
  
* Push

  ```commandline
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/tensorflow/unet:1.14.0_2010-02-22
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```
  
* Pull

  If image is available in aml-repo and you just want to use it, you can pull using following command and then [run](#run).

  ```commandline
  docker pull public.aml-repo.cms.waikato.ac.nz:443/tensorflow/unet:1.14.0_2010-02-22
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public.aml-repo.cms.waikato.ac.nz:443
  ```
  Then tag by running:
  
  ```commandline
  docker tag \
    public.aml-repo.cms.waikato.ac.nz:443/tensorflow/unet:1.14.0_2010-02-22 \
    tensorflow/unet:1.14.0_2010-02-22
  ```
  
* <a name="run">Run</a>

  ```commandline
  docker run --runtime=nvidia -v /local:/container -it tensorflow/unet:1.14.0_2010-02-22
  ```
  `/local:/container` maps a local disk directory into a directory inside the container
