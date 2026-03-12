FROM debian:bullseye

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
       git-core \
       wget \
       python3 \
       python3-dev \
       python3-venv \
       python3-pip \
       python3-numpy \
       python3-full \
       build-essential \
       libglib2.0-0 \
       libsm6 \
       libxrender-dev \
       libxext6 \
       libgl1 \
       libglx-mesa0 \
       libpq-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt

RUN python3 -m venv --system-site-packages /opt/yamnet
RUN /opt/yamnet/bin/pip install --no-cache-dir \
      tensorflow==2.20.0 \
      tf-keras==2.20.1 \
      tensorflow_hub
RUN /opt/yamnet/bin/pip install --no-cache-dir \
      scipy==1.15.3 \
      "simple-file-poller>=0.0.9" \
      redis \
      "redis-docker-harness==0.0.4"

COPY bash.bashrc /etc/bash.bashrc
COPY predict_*.py /opt/yamnet-extra/
COPY yamnet_* /usr/local/bin/

ENV PYTHONPATH=/opt/yamnet-extra

WORKDIR /workspace
