ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}tensorflow/tensorflow:1.11.0-devel-gpu-py3

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
       git-core \
       wget \
       build-essential \
       libglib2.0-0 \
       libsm6 \
       libxrender-dev \
       libxext6 \
       libgl1-mesa-glx \
       libpq-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone https://github.com/google-research/bert.git && \
    cd bert && \
    git reset --hard eedf5716ce1268e56f0a50264a88cafad334ac61

COPY bash.bashrc /etc/bash.bashrc

ENV PYTHONPATH=/opt/bert

WORKDIR /workspace

