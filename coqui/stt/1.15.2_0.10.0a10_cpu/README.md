# Coqui STT (CPU)

Allows speech-to-text using Tensorflow 1.5.2 and stt-tflite 0.10.0a10 (CPU).

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
    -it public.aml-repo.cms.waikato.ac.nz:443/tensorflow/tf_coqui_stt:1.15.2_0.10.0a10_cpu
  ```

* If need be, remove all containers and images from your system:

  ```bash
  docker stop $(docker ps -a -q) && docker rm $(docker ps -a -q) && docker system prune -a
  ```

### Docker hub

The image is also available from [Docker hub](https://hub.docker.com/u/waikatodatamining):

```
waikatodatamining/tf_coqui_stt:1.15.2_0.10.0a10_cpu
```


### Build local image

* Build the image from Docker file (from within /path_to/tensorflow/tf_coqui_stt/docker/1.15.2_0.10.0a10_cpu)

  ```bash
  docker build -t stt_cpu .
  ```

* Run the container

  ```bash
  docker run \
    -v /local/dir:/container/dir \
    -it stt_cpu
  ```

### Pre-built images

* Build

  ```bash
  docker build -t tensorflow/tf_coqui_stt:1.15.2_0.10.0a10_cpu .
  ```
  
* Tag

  ```bash
  docker tag \
    tensorflow/tf_coqui_stt:1.15.2_0.10.0a10_cpu \
    public-push.aml-repo.cms.waikato.ac.nz:443/tensorflow/tf_coqui_stt:1.15.2_0.10.0a10_cpu
  ```
  
* Push

  ```bash
  docker push public-push.aml-repo.cms.waikato.ac.nz:443/tensorflow/tf_coqui_stt:1.15.2_0.10.0a10_cpu
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```bash
  docker login public-push.aml-repo.cms.waikato.ac.nz:443
  ```
  
* Pull

  If image is available in aml-repo and you just want to use it, you can pull using following command and then [run](#run).

  ```bash
  docker pull public.aml-repo.cms.waikato.ac.nz:443/tensorflow/tf_coqui_stt:1.15.2_0.10.0a10_cpu
  ```
  If error "no basic auth credentials" occurs, then run (enter username/password when prompted):
  
  ```bash
  docker login public.aml-repo.cms.waikato.ac.nz:443
  ```
  Then tag by running:
  
  ```bash
  docker tag \
    public.aml-repo.cms.waikato.ac.nz:443/tensorflow/tf_coqui_stt:1.15.2_0.10.0a10_cpu \
    tensorflow/tf_coqui_stt:1.15.2_0.10.0a10_cpu
  ```

* <a name="run">Run</a>

  ```bash
  docker run \
    -v /local/dir:/container/dir \
    -it tensorflow/tf_coqui_stt:1.15.2_0.10.0a10_cpu
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

* `stt_transcribe_single` - generating a transcription on a single audio file with a tflite model (calls `/opt/coqui_ext/transcribe_single.py`)
* `stt_transcribe_poll` - processes audio files via file-polling with a tflite model (calls `/opt/coqui_ext/transcribe_poll.py`)
* `stt_transcribe_redis` - processes audios file with a tflite model using [Redis](https://redis.io/) (calls `/opt/coqui_ext/transcribe_redis.py`)
