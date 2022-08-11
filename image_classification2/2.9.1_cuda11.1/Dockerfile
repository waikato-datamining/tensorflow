ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}tensorflow/tensorflow:2.9.1-gpu

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "tensorflow-hub[make_image_classifier]~=0.6"
RUN pip install --no-cache-dir python-image-complete "simple-file-poller>=0.0.9" && \
    pip install --no-cache-dir redis "opex==0.0.1" "redis-docker-harness==0.0.1"

COPY bash.bashrc /etc/bash.bashrc

COPY label_image.py /opt/tfimgcls/
COPY predict_poll.py /opt/tfimgcls/
COPY predict_redis.py /opt/tfimgcls/
COPY predict_utils.py /opt/tfimgcls/
COPY test_image_redis.py /opt/tfimgcls/

COPY label_image /usr/bin
COPY predict_poll /usr/bin
COPY predict_redis /usr/bin
COPY test_image_redis /usr/bin

