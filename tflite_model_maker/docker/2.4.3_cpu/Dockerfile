ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}ubuntu:bionic

RUN apt-get update && \
        apt-get upgrade -y && \
        apt-get -y install \
          libglib2.0-0 \
          libsm6 \
          libxrender-dev \
          libxext6 \
          libsndfile1 \
          python3.7 \
          python3-setuptools \
          python3-pip && \
        pip3 --no-cache-dir install --upgrade pip setuptools && \
        rm -rf /var/lib/apt/lists/* && \
        ln -s /usr/bin/pip3 /usr/bin/pip && \
        ln -s /usr/bin/python3 /usr/bin/python && \
        pip install "numpy<1.20.0" && \
        pip install wai.pycocotools && \
        pip install "tflite-model-maker>=0.3.0,<0.3.2" "tensorflow>2.4.0,<2.5.0" pyyaml opex python-image-complete "simple-file-poller>=0.0.9" redis "redis-docker-harness==0.0.1" && \
        pip install "wai.tflite_model_maker==0.0.4" && \
        rm -Rf /root/.cache/pip && \
        rm -rf /var/lib/apt/lists/*

# https://stackoverflow.com/questions/59290386/runtimeerror-at-cannot-cache-function-shear-dense-no-locator-available-fo
ENV NUMBA_CACHE_DIR=/tmp

COPY bash.bashrc /etc/bash.bashrc

WORKDIR /
