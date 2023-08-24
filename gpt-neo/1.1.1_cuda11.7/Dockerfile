ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}tensorflow/tensorflow:2.5.1-gpu

# update NVIDIA repo key
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
ARG distro=ubuntu1804
ARG arch=x86_64
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/3bf863cc.pub

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt                                
RUN git clone https://github.com/EleutherAI/gpt-neo.git && \
    cd gpt-neo && \
    git checkout v1.1.1 && \
    pip install ortools==9.0.9048 && \
    pip install -r requirements.txt

COPY bash.bashrc /etc/bash.bashrc
COPY gptneo* /usr/local/bin/

ENV PYTHONPATH=/opt/gpt-neo

