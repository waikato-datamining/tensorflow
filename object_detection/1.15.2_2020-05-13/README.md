# Object Detection framework

Allows processing of images with Tensorflow's Object Detection framework, using Tensorflow 1.15.2.

## Version

Object Detection framework github repo hash:

```
8518d053936aaf30afb9ed0a4ea01baddca5bd17
```

and timestamp:

```
13 May 2020, 6:41am GMT
```

## Docker

### Quick start

* Log into registry using *public* credentials:

  ```commandline
  docker login -u public -p public public.aml-repo.cms.waikato.ac.nz:443 
  ```

* Pull and run image (adjust volume mappings `-v`):

  ```commandline
  docker run --gpus=all \
    -v /local/dir:/container/dir \
    -it public.aml-repo.cms.waikato.ac.nz:443/tensorflow/object_detection:1.15.2_2020-05-13
  ```

  **NB:** For docker versions older than 19.03 (`docker version`), use `--runtime=nvidia` instead of `--gpus=all`.

* If need be, remove all containers and images from your system:

  ```commandline
  docker stop $(docker ps -a -q) && docker rm $(docker ps -a -q) && docker system prune -a
  ```


### Build local image

* Build image `tf` from Docker file (from within /path_to/tensorflow/object_detection/1.15.2_2020-05-13)

  ```commandline
  docker build -t tf .
  ```
  
* Run image `tf` in interactive mode (i.e., using `bash`) as container `tf_container`

  ```commandline
  docker run --gpus=all --name tf_container -ti -v \
    /local/dir:/container/dir \
    tf bash
  ```

### Pre-built images

* Build

  ```commandline
  docker build -t tensorflow/object_detection:1.15.2_2020-05-13 .
  ```
  
* Tag

  ```commandline
  docker tag \
    tensorflow/object_detection:1.15.2_2020-05-13 \
    public-push.aml-repo.cms.waikato.ac.nz:443/tensorflow/object_detection:1.15.2_2020-05-13
  ```
  
* Push

  ```commandline
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/tensorflow/object_detection:1.15.2_2020-05-13
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```
  
* Pull

  If image is available in aml-repo and you just want to use it, you can pull using following command and then [run](#run).

  ```commandline
  docker pull public.aml-repo.cms.waikato.ac.nz:443/tensorflow/object_detection:1.15.2_2020-05-13
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public.aml-repo.cms.waikato.ac.nz:443
  ```
  Then tag by running:
  
  ```commandline
  docker tag \
    public.aml-repo.cms.waikato.ac.nz:443/tensorflow/object_detection:1.15.2_2020-05-13 \
    tensorflow/object_detection:1.15.2_2020-05-13
  ```
  
* <a name="run">Run</a>

  ```commandline
  docker run --gpus=all -v /local/dir:/container/dir -it tensorflow/object_detection:1.15.2_2020-05-13
  ```
  `/local/dir:/container/dir` maps a local disk directory into a directory inside the container

### Usage

* Generate tfrecords, e.g., from ADAMS annotations (using [wai.annotations](https://github.com/waikato-ufdl/wai-annotations))

  ```commandline
  convert-annotations \
    adams -i /path_to/images_and_reports_directory \
    tfrecords -o /path_to/name_of_output_file.records \
    -p /path_to/name_of_output_labels_file.pbtxt
  ```
  Run with `-h/--help` for all available options.
  Above command need to run twice, once for training set and again for validation set.

  In case of generating output for Mask RCNN, you need to use global option `-f mask`.

* Update the config file (data augmentation: [1](https://stackoverflow.com/a/46901051/4698227), [2](https://github.com/tensorflow/models/blob/master/research/object_detection/core/preprocessor.py), [3](https://github.com/tensorflow/models/blob/master/research/object_detection/builders/preprocessor_builder_test.py)) and then start training:

  ```commandline
  objdet_train --pipeline_config_path=/path_to/your_data.config \
    --model_dir=/path_to/your_data/output --num_train_steps=50000 \
    --sample_1_of_n_eval_examples=1 --alsologtostderr
  ```

  For training Mask RCNN, use this script (number of steps must be defined in the .config file, see [here](https://github.com/vijaydwivedi75/Custom-Mask-RCNN_TF/blob/master/mask_rcnn_inception_v2_coco.config) for an example):

  ```commandline
  objdet_train_legacy --train_dir=/path/to/your/output \
    --pipeline_config_path=/path/to/config/mask_rcnn_inception_v2_coco.config
  ```

* Export inference graph

  ```commandline
  objdet_export --input_type image_tensor --pipeline_config_path /path_to/your_data.config \
    --trained_checkpoint_prefix /path_to/your_data/output/model.ckpt-50000 \
    --output_directory /path_to/your_data/output/exported_graphs
  ```

* Predict and produce CSV files

  ```commandline
  objdet_predict --graph /path_to/your_data/output/exported_graphs/frozen_inference_graph.pb \
    --labels /path_to/your_data_label_map.pbtxt --prediction_in /path_to/your_data/test_images/ \
    --prediction_out /path_to/your_data/output/results --score 0.1
  ```
  Run with -h for all available options.

  In case of outputting predictions for Mask RCNN, you also need to specify the
  following options:

  * `--output_polygons`
  * `--mask_threshold` - if using another threshold than the default of 0.1
  * `--mask_nth` - use every nth row/col of mask to speed up computation of polygon
  * `--output_minrect`

## Background images

In order to use manually curated background images, you have to add the following 
`hard_example_miner` section to your `pipeline.config`:

```protobuf
model {
  faster_rcnn {
    ...
    hard_example_miner {
      num_hard_examples: 0
      use_negative_images_only: true
    }
  }
  ...
}
```

And, when generating the tfrecords from ADAMS annotations, the background images 
need to get added to the training data with corresponding (empty) `.report` files 
([wai.annotations](https://github.com/waikato-ufdl/wai-annotations) iterates 
via the `.report` files).


## Permissions

When running the docker container as regular use, you will want to set the correct
user and group on the files generated by the container (aka the user:group launching
the container):

```commandline
docker run -u $(id -u):$(id -g) ...
```
