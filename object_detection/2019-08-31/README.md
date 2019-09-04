# Object Detection framework

Allows processing of images with Tensorflow's Object Detection framework. 

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
  /path_to/local_disk/contianing_data:/path_to/mount/inside/docker_container tf bash
  ```

* Update the config file and then start training (assuming your data and tfrecords are ready)

  ```commandline
  python object_detection/model_main.py --pipeline_config_path=/path_to/your_data.config \
  --model_dir=/path_to/your_data/output --num_train_steps=50000 \
  --sample_1_of_n_eval_examples=1 --alsologtostderr
  ```

* Export frozen_inference_graph.pb

  ```commandline
  python object_detection/export_inference_graph.py --input_type image_tensor \
  --pipeline_config_path /path_to/your_data.config \
  --trained_checkpoint_prefix /path_to/your_data/output/model.ckpt-50000 \
  --output_directory /path_to/your_data/output/exported_graphs
  ```

* Predict and produce csv files (from within /opt/tensorflow/object_detection/2019-08-31)

  ```commandline
  python predict.py --graph /path_to/your_data/output/exported_graphs/frozen_inference_graph.pb \
  --labels /path_to/your_data_label_map.pbtxt --prediction_in /path_to/your_data/test_images/ \
  --prediction_out /path_to/your_data/output/results --score 0.1 --num_imgs 3 --num_classes 2
  ```

* Execute the `predict.py` script with `-h` to see the help screen.

## Docker Image Push to aml-repo

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