# tensorflow 2 image classification (CUDA 11.1)

Allows classifying images with Tensorflow's image classification capabilities, using Tensorflow 2.9.1 (CUDA 11.1).

## Docker

### Quick start

* Log into registry using *public* credentials:

  ```bash
  docker login -u public -p public public.aml-repo.cms.waikato.ac.nz:443 
  ```

* Pull and run image (adjust volume mappings `-v`):

  ```bash
  docker run \
    --gpus=all \
    -v /local/dir:/container/dir \
    -it public.aml-repo.cms.waikato.ac.nz:443/tensorflow/tf_image_classification2:2.9.1_cuda11.1
  ```

* If need be, remove all containers and images from your system:

  ```bash
  docker stop $(docker ps -a -q) && docker rm $(docker ps -a -q) && docker system prune -a
  ```

### Docker hub

The image is also available from [Docker hub](https://hub.docker.com/u/waikatodatamining):

```
waikatodatamining/tf_image_classification2:2.9.1_cuda11.1
```


### Build local image

* Build the image from Docker file (from within /path_to/tensorflow/tf_image_classification2/docker/2.9.1_cuda11.1)

  ```bash
  docker build -t imgcls2 .
  ```

* Run the container

  ```bash
  docker run \
    --gpus=all \
    -v /local/dir:/container/dir \
    -it imgcls2
  ```

### Pre-built images

* Build

  ```bash
  docker build -t tensorflow/tf_image_classification2:2.9.1_cuda11.1 .
  ```
  
* Tag

  ```bash
  docker tag \
    tensorflow/tf_image_classification2:2.9.1_cuda11.1 \
    public-push.aml-repo.cms.waikato.ac.nz:443/tensorflow/tf_image_classification2:2.9.1_cuda11.1
  ```
  
* Push

  ```bash
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/tensorflow/tf_image_classification2:2.9.1_cuda11.1
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```bash
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```
  
* Pull

  If image is available in aml-repo and you just want to use it, you can pull using following command and then [run](#run).

  ```bash
  docker pull public.aml-repo.cms.waikato.ac.nz:443/tensorflow/tf_image_classification2:2.9.1_cuda11.1
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```bash
  docker login public.aml-repo.cms.waikato.ac.nz:443
  ```
  Then tag by running:
  
  ```bash
  docker tag \
    public.aml-repo.cms.waikato.ac.nz:443/tensorflow/tf_image_classification2:2.9.1_cuda11.1 \
    tensorflow/tf_image_classification2:2.9.1_cuda11.1
  ```

* <a name="run">Run</a>

  ```bash
  docker run \ 
    --gpus=all \
    -v /local/dir:/container/dir \
    -it tensorflow/tf_image_classification2:2.9.1_cuda11.1
  ```
  `/local/dir:/container/dir` maps a local disk directory into a directory inside the container


## Permissions

When running the docker container as regular use, you will want to set the correct
user and group on the files generated by the container (aka the user:group launching
the container):

```bash
docker run -u $(id -u):$(id -g) ...
```


## Usage

The following command-line tools are available:

* `make_image_classifier` - for training a trained model on a dataset
* `label_image` - labeling a single image
* `predict_poll` - for batch or continuous predictions
* `predict_redis` - for predictions via Redis backend (use `--net=host` to get access to host's Redis instance)
* `test_image_redis` - helper tool for broadcasting an image via Redis
