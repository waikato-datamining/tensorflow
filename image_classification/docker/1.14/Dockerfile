# Ubuntu 18.04, CUDA 10.0, Python 3.6
ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}tensorflow/tensorflow:1.14.0-gpu-py3

# update NVIDIA repo key
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
ARG distro=ubuntu1804
ARG arch=x86_64
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/3bf863cc.pub

# Note: don't run "apt-get upgrade -y" as the update will result in the CUDA/cuDNN backend not initializing properly??
RUN apt-get update && \
        rm -rf /var/lib/apt/lists/* && \
        pip install Cython && \
        pip install tensorflow-hub==0.9.0 && \
        pip install wai.tfimageclass==0.0.15 && \
        pip uninstall -y numpy && \
        pip install "numpy<=1.17.4" && \
        rm -Rf /root/.cache/pip

COPY bash.bashrc /etc/bash.bashrc

WORKDIR /
