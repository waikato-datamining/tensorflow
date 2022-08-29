ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}tensorflow/tensorflow:1.15.2-py3

RUN pip install --no-cache-dir "stt==1.3.0" && \
    pip install --no-cache-dir "python-image-complete" "simple-file-poller>=0.0.9" && \
    pip install --no-cache-dir redis "redis-docker-harness==0.0.1"

COPY bash.bashrc /etc/bash.bashrc
COPY stt_* /usr/bin/
COPY transcribe* /opt/coqui_ext/

ENV PYTHONPATH=$PYTHONPATH:/opt/coqui_ext

WORKDIR /workspace
