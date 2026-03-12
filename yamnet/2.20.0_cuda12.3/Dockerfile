FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

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

ENV PATH=$PATH:/usr/local/cuda-12.3/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.3/lib64
ENV CUDA_HOME=/usr/local/cuda-12.3

WORKDIR /opt

RUN python3 -m venv --system-site-packages /opt/yamnet
RUN /opt/yamnet/bin/pip install --no-cache-dir \
      tensorflow[and-cuda]==2.20.0 \
      tf-keras==2.20.1 \
      tensorflow_hub
RUN /opt/yamnet/bin/pip install --no-cache-dir \
      env-doctor

COPY bash.bashrc /etc/bash.bashrc
WORKDIR /workspace

