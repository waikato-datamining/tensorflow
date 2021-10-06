ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}tensorflow/tensorflow:2.4.3-gpu

# Note: don't run "apt-get upgrade -y" as the update will result in the CUDA/cuDNN backend not initializing properly??
RUN apt-get update && \
        rm -rf /var/lib/apt/lists/* && \
        pip install "numpy<1.20.0" && \
        pip install wai.pycocotools && \
        pip install "tflite-model-maker>=0.3.0,<0.3.2" "tensorflow>2.4.0,<2.5.0" pyyaml opex python-image-complete "simple-file-poller>=0.0.9" redis "redis-docker-harness==0.0.1" && \
        rm -Rf /root/.cache/pip

COPY bash.bashrc /etc/bash.bashrc

WORKDIR /
