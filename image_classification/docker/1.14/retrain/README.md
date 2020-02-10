# Retraining a pre-trained model

Performs a training run on a pre-trained model.

Uses `tfic-retrain` of the [wai.tfimageclass](https://pypi.org/project/wai.tfimageclass/)
library.

## Docker

### Build local image

* Build the image from Docker file (from within /path_to/tensorflow/image_classification/docker/1.14/retrain)

  ```commandline
  docker build -t tfic_retrain .
  ```

* Run the container

  ```commandline
  docker run --runtime=nvidia -v /local:/container -it tfic_retrain
  ```

### Pre-built images

* Build

  ```commandline
  docker build -t tensorflow/image_classification:1.14_retrain .
  ```
  
* Tag

  ```commandline
  docker tag \
    tensorflow/image_classification:1.14 \
    public-push.aml-repo.cms.waikato.ac.nz:443/tensorflow/image_classification:1.14_retrain
  ```
  
* Push

  ```commandline
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/tensorflow/image_classification:1.14_retrain
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```
  
* Pull

  If image is available in aml-repo and you just want to use it, you can pull using following command and then [run](#run).

  ```commandline
  docker pull public.aml-repo.cms.waikato.ac.nz:443/tensorflow/image_classification:1.14_retrain
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public.aml-repo.cms.waikato.ac.nz:443
  ```
  Then tag by running:
  
  ```commandline
  docker tag \
    public.aml-repo.cms.waikato.ac.nz:443/tensorflow/image_classification:1.14_retrain \
    tensorflow/image_classification:1.14_retrain
  ```

* <a name="run">Run</a>

  ```commandline
  docker run --runtime=nvidia -v /local:/container -it tensorflow/image_classification:1.14_retrain \
    --image_dir /path/to/data/ \
    --image_lists_dir /path/to/output/ \
    --output_graph /path/to/output/output_graph.pb \
    --output_labels /path/to/output/output_labels.txt \
    --checkpoint_path /path/to/output/retrain_checkpoint \
    --saved_model_dir /path/to/output/saved_model \
    --bottleneck_dir /path/to/output/bottleneck \
    --intermediate_output_graphs_dir /path/to/output/intermediate_graph \
    --summaries_dir /path/to/output/retrain_logs \
    --tfhub_module "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/3" \
    --testing_percentage 20 \
    --validation_percentage 20 \
    --training_steps 2000
  ```
  "/local:/container" maps a local disk directory into a directory inside the container

