FROM ghcr.io/coqui-ai/stt-train:v1.4.0

RUN pip install --no-cache-dir "stt==1.4.0" && \
    pip install --no-cache-dir "simple-file-poller>=0.0.9" && \
    pip install --no-cache-dir redis "redis-docker-harness==0.0.1"

ENV BASH_ENV=
COPY bash.bashrc /etc/bash.bashrc
COPY stt_* /usr/bin/
COPY alphabet* /opt/coqui_ext/
COPY transcribe* /opt/coqui_ext/
COPY test_audio* /opt/coqui_ext/

ENV PYTHONPATH=$PYTHONPATH:/opt/coqui_ext

WORKDIR /workspace

