# Ubuntu 18.04, CUDA 10.0, Python 3.6
ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}tensorflow/tensorflow:1.14.0-gpu-py3

# Note: don't run "apt-get upgrade -y" as the update will result in the CUDA/cuDNN backend not initializing properly??
RUN apt-get update && \
        rm -rf /var/lib/apt/lists/* && \
        pip install Cython && \
        pip install wai.tfimageclass && \
        rm -Rf /root/.cache/pip

WORKDIR /
