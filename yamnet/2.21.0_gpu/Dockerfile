FROM tensorflow/tensorflow:2.21.0-gpu

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

COPY bash.bashrc /etc/bash.bashrc
WORKDIR /workspace
