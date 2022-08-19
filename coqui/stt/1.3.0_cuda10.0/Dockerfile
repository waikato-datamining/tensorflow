ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# update NVIDIA repo key
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
ARG distro=ubuntu1804
ARG arch=x86_64
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/3bf863cc.pub

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip python3-venv wget \
    libopus0 libvorbisfile3 libopusfile0 libsndfile1 sox libsox-fmt-mp3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/coqui-ai/STT.git /coqui-stt && \
    cd /coqui-stt && \
    git checkout tags/v1.3.0

WORKDIR /coqui-stt

ENV VIRTUAL_ENV=/coqui-stt-venv
RUN python3 -m venv $VIRTUAL_ENV
RUN . $VIRTUAL_ENV/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e . && \
    pip uninstall -y tensorflow && \
    pip install --no-cache-dir 'tensorflow-gpu==1.15.4' 'torch==1.4.0'

