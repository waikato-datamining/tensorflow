ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}nvidia/cuda:10.0-base-ubuntu18.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends git-core wget libglib2.0-0 libsm6 libxrender-dev libxext6 python3 python3-pip ffmpeg && \
    ln -s /usr/bin/pip3 /usr/bin/pip && \
    rm -rf /var/lib/apt/lists/* && \
    pip install Cython && \
    pip install "numpy<1.18.0" && \
    pip install "deepspeech-gpu==0.7.4" pydub && \
    rm -Rf /root/.cache/pip

COPY bash.bashrc /etc/bash.bashrc

WORKDIR /
