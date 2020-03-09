# Batch processing images

# Generating statistics

Performs batch processing of/polling for images, taking images from the input directory
and placing the predictions in the output directory.

Uses `tfic-poll` of the [wai.tfimageclass](https://pypi.org/project/wai.tfimageclass/)
library.

## Docker

### Build local image

* Build the image from Docker file (from within /path_to/tensorflow/image_classification/docker/1.14/label_image)

  ```commandline
  docker build -t tfic_poll .
  ```

* Run the container

  ```commandline
  docker run --runtime=nvidia -v /local:/container -it tfic_poll
  ```

### Pre-built images

* Build

  ```commandline
  docker build -t tensorflow/image_classification:1.14_poll .
  ```
  
* Tag

  ```commandline
  docker tag \
    tensorflow/image_classification:1.14 \
    public-push.aml-repo.cms.waikato.ac.nz:443/tensorflow/image_classification_poll:1.14
  ```
  
* Push

  ```commandline
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/tensorflow/image_classification_poll:1.14
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```
  
* Pull

  If image is available in aml-repo and you just want to use it, you can pull using following command and then [run](#run).

  ```commandline
  docker pull public.aml-repo.cms.waikato.ac.nz:443/tensorflow/image_classification_poll:1.14
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public.aml-repo.cms.waikato.ac.nz:443
  ```
  Then tag by running:
  
  ```commandline
  docker tag \
    public.aml-repo.cms.waikato.ac.nz:443/tensorflow/image_classification_poll:1.14 \
    tensorflow/image_classification_poll:1.14
  ```

* <a name="run">Run</a>

  ```commandline
  docker run --runtime=nvidia -v /local:/container -it tensorflow/image_classification_poll:1.14 \
    --image /path/to/image.jpg \
    --graph /path/to/output_graph.pb \
    --labels /path/to/output_labels.txt \
    --input_width 224 --input_height 224 \
    --in_dir /path/to/predictions/in \ 
    --out_dir /path/to/predictions/out 
  ```
  `/local:/container` maps a local disk directory into a directory inside the container

