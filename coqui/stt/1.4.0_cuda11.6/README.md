# Coqui STT (CUDA 11.6)

Allows speech-to-text using Tensorflow 1.15.5 (CUDA 11.6).

## Docker

### Quick start

* Log into registry using *public* credentials:

  ```bash
  docker login -u public -p public public.aml-repo.cms.waikato.ac.nz:443 
  ```

* Pull and run image (adjust volume mappings `-v`):

  ```bash
  docker run \
    --gpu=all \
    --shm-size=1g \
    -v /local/dir:/container/dir \
    -it public.aml-repo.cms.waikato.ac.nz:443/tensorflow/tf_coqui_stt:1.4.0_cuda11.6
  ```

* If need be, remove all containers and images from your system:

  ```bash
  docker stop $(docker ps -a -q) && docker rm $(docker ps -a -q) && docker system prune -a
  ```

### Docker hub

The image is also available from [Docker hub](https://hub.docker.com/u/waikatodatamining):

```
waikatodatamining/tf_coqui_stt:1.4.0_cuda11.6
```


### Build local image

* Build the image from Docker file (from within /path_to/tensorflow/tf_coqui_stt/docker/1.4.0_cuda11.6)

  ```bash
  docker build -t stt .
  ```

* Run the container

  ```bash
  docker run \
    --gpus=all \
    --shm-size=1g \
    -v /local/dir:/container/dir \
    -it stt
  ```

### Pre-built images

* Build

  ```bash
  docker build -t tensorflow/tf_coqui_stt:1.4.0_cuda11.6 .
  ```
  
* Tag

  ```bash
  docker tag \
    tensorflow/tf_coqui_stt:1.4.0_cuda11.6 \
    public-push.aml-repo.cms.waikato.ac.nz:443/tensorflow/tf_coqui_stt:1.4.0_cuda11.6
  ```
  
* Push

  ```bash
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/tensorflow/tf_coqui_stt:1.4.0_cuda11.6
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```bash
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```
  
* Pull

  If image is available in aml-repo and you just want to use it, you can pull using following command and then [run](#run).

  ```bash
  docker pull public.aml-repo.cms.waikato.ac.nz:443/tensorflow/tf_coqui_stt:1.4.0_cuda11.6
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```bash
  docker login public.aml-repo.cms.waikato.ac.nz:443
  ```
  Then tag by running:
  
  ```bash
  docker tag \
    public.aml-repo.cms.waikato.ac.nz:443/tensorflow/tf_coqui_stt:1.4.0_cuda11.6 \
    tensorflow/tf_coqui_stt:1.4.0_cuda11.6
  ```

* <a name="run">Run</a>

  ```bash
  docker run \
    --gpus=all \
    --shm-size=1g \
    -v /local/dir:/container/dir \
    -it tensorflow/tf_coqui_stt:1.4.0_cuda11.6
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

* `stt_alphabet` - generates an alphabet.txt file from one or more transcript CSV files (calls `/opt/coqui_ext/alphabet.py`)
* `stt_train` - for training a STT model (calls `python -m coqui_stt_training.train`)
* `stt_eval` - for evaluating a trained STT model (calls `python -m coqui_stt_training.evaluate`)
* `stt_export` - for exporting a trained STT model to tflite (calls `python -m coqui_stt_training.export`)
* `stt_infer` - for performing inference with a trained STT model (calls `python -m coqui_stt_training.training_graph_inference`)
* `stt_transcribe_single` - generating a transcription on a single audio file with a tflite model (calls `/opt/coqui_ext/transcribe_single.py`)


## Troubleshooting

* To avoid the training to hang and never exit once it finishes, use the `--skip_batch_test true` option