# Object Detection framework

Allows processing of images with Tensorflow's Object Detection framework, using Tensorflow 1.14.0.

# Version

Object Detection framework github repo hash:

```
b9ef963d1e84da0bb9c0a6039457c39a699ea149
```

and timestamp:

```
Fri Aug 30 14:39:49 2019 -0700
```

COCO API github repo hash:

```
636becdc73d54283b3aac6d4ec363cffbb6f9b20
```

## Installation

* install virtual environment

  ```commandline
  virtualenv -p /usr/bin/python2.7 venv
  ```

* install dependencies

  ```commandline
  ./venv/bin/pip install -r requirements.txt 
  ```

* install tensorflow

  * with GPU
  
    ```commandline
    ./venv/bin/pip install tensorflow-gpu
    ```
    
  * CPU only
  
    ```commandline
    ./venv/bin/pip install tensorflow
    ```
    
* install object detection framework ([instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md))

  ```commandline
  git clone https://github.com/tensorflow/models
  ```

* you need to add the directory above the `object_detection` one to the `PYTHONPATH`
  environment variable.
  
  ```commandline
  export PYTHONPATH=$PYTHONPATH:/some/where/models/research
  ```
## Installation & Usage on Linux with Docker

* Build the image from Docker file (from within /path_to/tensorflow/object_detection/2019-08-31)

  ```commandline
  sudo docker build -t tf .
  ```
  
* Run the container

  ```commandline
  sudo docker run --runtime=nvidia --name tf_container -ti -v \
    /path_to/local_disk/containing_data:/path_to/mount/inside/docker_container tf bash
  ```

* Generate tfrecords (see also [wai.tfrecords](https://github.com/waikato-datamining/tensorflow/tree/master/tfrecords))

  ```commandline
  objdet_convert -i /path_to/images_and_reports_directory \
    -o /path_to/name_of_output_file.tfrecords -s number_of_shards \
    -p /path_to/name_of_output_labels_file.pbtxt -m mapping_old_label=new_label \
    -r regexp_for_using_only_subset_of_labels
  ```
  Run with `-h/--help` for all available options.
  Above command need to run twice, once for training set and again for validation set.

* Update the config file (data augmentation: [1](https://stackoverflow.com/a/46901051/4698227), [2](https://github.com/tensorflow/models/blob/master/research/object_detection/core/preprocessor.py), [3](https://github.com/tensorflow/models/blob/master/research/object_detection/builders/preprocessor_builder_test.py)) and then start training:

  ```commandline
  objdet_train --pipeline_config_path=/path_to/your_data.config \
    --model_dir=/path_to/your_data/output --num_train_steps=50000 \
    --sample_1_of_n_eval_examples=1 --alsologtostderr
  ```

* Export frozen_inference_graph.pb

  ```commandline
  objdet_export --input_type image_tensor --pipeline_config_path /path_to/your_data.config \
    --trained_checkpoint_prefix /path_to/your_data/output/model.ckpt-50000 \
    --output_directory /path_to/your_data/output/exported_graphs
  ```

* Predict and produce csv files

  ```commandline
  objdet_predict --graph /path_to/your_data/output/exported_graphs/frozen_inference_graph.pb \
    --labels /path_to/your_data_label_map.pbtxt --prediction_in /path_to/your_data/test_images/ \
    --prediction_out /path_to/your_data/output/results --score 0.1 --num_imgs 3 --num_classes 1
  ```
  Run with -h for all available options.

## Docker Image in aml-repo

* Build

  ```commandline
  docker build -t tensorflow/object_detection:2019-08-31 .
  ```
  
* Tag

  ```commandline
  docker tag \
    tensorflow/object_detection:2019-08-31 \
    public-push.aml-repo.cms.waikato.ac.nz:443/tensorflow/object_detection:2019-08-31
  ```
  
* Push

  ```commandline
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/tensorflow/object_detection:2019-08-31
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```
  
* Pull

  If image is available in aml-repo and you just want to use it, you can pull using following command and then [run](#run).

  ```commandline
  docker pull public.aml-repo.cms.waikato.ac.nz:443/tensorflow/object_detection:2019-08-31
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```commandline
  docker login public.aml-repo.cms.waikato.ac.nz:443
  ```
  Then tag by running:
  
  ```commandline
  docker tag \
    public.aml-repo.cms.waikato.ac.nz:443/tensorflow/object_detection:2019-08-31 \
    tensorflow/object_detection:2019-08-31
  ```
  
* <a name="run">Run</a>

  ```commandline
  docker run --runtime=nvidia -v /local:/container -it tensorflow/object_detection:2019-08-31
  ```
  "/local:/container" maps a local disk directory into a directory inside the container
