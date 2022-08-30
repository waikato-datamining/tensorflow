ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3.7-full libpython3.7 git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py && \
    python3.7 /tmp/get-pip.py && \
    rm /tmp/get-pip.py && \
    python3.7 -m pip install --no-cache-dir "tensorflow==1.15.2" && \
    python3.7 -m pip install --no-cache-dir "stt-tflite==0.10.0a10" && \
    python3.7 -m pip install --no-cache-dir "simple-file-poller>=0.0.9" && \
    python3.7 -m pip install --no-cache-dir redis "redis-docker-harness==0.0.1"

COPY bash.bashrc /etc/bash.bashrc
COPY stt_* /usr/bin/
COPY transcribe* /opt/coqui_ext/
COPY test_audio* /opt/coqui_ext/

ENV PYTHONPATH=$PYTHONPATH:/opt/coqui_ext

WORKDIR /workspace

