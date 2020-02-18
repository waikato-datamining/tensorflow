# Exporting graphs with Object Detection framework

Allows processing of images with Tensorflow's Object Detection framework, using Tensorflow 1.15.2.

## Version

Object Detection framework github repo hash:

```
b9ef963d1e84da0bb9c0a6039457c39a699ea149
```

and timestamp:

```
Fri Aug 30 14:39:49 2019 -0700
```

## Docker

### Build local image

* Build the image from Docker file (from within /path_to/tensorflow/object_detection/1.15.2_2019-08-31/export)

  ```commandline
  docker build -t tf_export .
  ```

* Run the container

  ```commandline
  docker run --runtime=nvidia -ti -v \
    /path_to/local_disk/containing_data:/path_to/mount/inside/docker_container tf_export \
    --input_type image_tensor --pipeline_config_path /path_to/your_data.config \
    --trained_checkpoint_prefix /path_to/your_data/output/model.ckpt-50000 \
    --output_directory /path_to/your_data/output/exported_graphs
  ```

### Pre-built images

* Build

  ```commandline
  docker build -t tensorflow/object_detection:1.15.2_2019-08-31_export .
  ```
  
* Tag

  ```commandline
  docker tag \
    tensorflow/object_detection:1.15.2_2019-08-31_export \
    public-push.aml-repo.cms.waikato.ac.nz:443/tensorflow/object_detection:1.15.2_2019-08-31_export
  ```
  
* Push

  ```commandline
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/tensorflow/object_detection:1.15.2_2019-08-31_export
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```
  
* Pull

  If image is available in aml-repo and you just want to use it, you can pull using following command and then [run](#run).

  ```commandline
  docker pull public.aml-repo.cms.waikato.ac.nz:443/tensorflow/object_detection:1.15.2_2019-08-31_export
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public.aml-repo.cms.waikato.ac.nz:443
  ```
  Then tag by running:
  
  ```commandline
  docker tag \
    public.aml-repo.cms.waikato.ac.nz:443/tensorflow/object_detection:1.15.2_2019-08-31_export \
    tensorflow/object_detection:1.15.2_2019-08-31_export
  ```

* <a name="run">Run</a>

  ```commandline
  docker run --runtime=nvidia -v /local:/container -it tensorflow/object_detection:1.15.2_2019-08-31_export \
    --input_type image_tensor --pipeline_config_path /path_to/your_data.config \
    --trained_checkpoint_prefix /path_to/your_data/output/model.ckpt-50000 \
    --output_directory /path_to/your_data/output/exported_graphs
  ```
  "/local:/container" maps a local disk directory into a directory inside the container

