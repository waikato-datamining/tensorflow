# based on:
# https://raw.githubusercontent.com/mozilla/DeepSpeech/v0.7.4/Dockerfile.train.tmpl

ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}tensorflow/tensorflow:1.15.2-gpu-py3
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        apt-utils \
        bash-completion \
        build-essential \
        curl \
        git \
        git-lfs \
        libbz2-dev \
        locales \
        python3-venv \
        unzip \
        wget \
        libglib2.0-0 \
        libsm6 \
        libxrender-dev \
        libxext6 \
        python3 \
        python3-pip \
        ffmpeg \
        sox \
        libsox-fmt-all && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/pip3 /usr/bin/pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -Rf /root/.cache/pip

# We need to remove it because it's breaking deepspeech install later with
# weird errors about setuptools
RUN apt-get update && \
    apt-get purge -y python3-xdg && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies for audio augmentation
RUN apt-get update && \
    apt-get install -y --no-install-recommends libopus0 libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /
RUN git lfs install
RUN git clone https://github.com/mozilla/DeepSpeech.git

WORKDIR /DeepSpeech
RUN git fetch && git fetch --tags && git checkout tags/v0.7.4

# Build CTC decoder first, to avoid clashes on incompatible versions upgrades
RUN cd native_client/ctcdecode && make NUM_PROCESSES=$(nproc) bindings
RUN pip3 install --upgrade native_client/ctcdecode/dist/*.whl && \
    rm -Rf /root/.cache/pip

# Prepare deps
RUN pip3 install --upgrade pip==20.0.2 wheel==0.34.2 setuptools==46.1.3 && \
    rm -Rf /root/.cache/pip

# Install DeepSpeech
#  - No need for the decoder since we did it earlier
#  - There is already correct TensorFlow GPU installed on the base image,
#    we don't want to break that
RUN DS_NODECODER=y DS_NOTENSORFLOW=y pip3 install --upgrade -e .

RUN ./bin/run-ldc93s1.sh

# install DeepSpeech client
RUN pip install deepspeech-gpu==0.7.4 && \
    rm -Rf /root/.cache/pip

# copy scripts
COPY bash.bashrc /etc/bash.bashrc
COPY pause_splitter.py /opt/deepspeech/pause_splitter.py
COPY predict.py /opt/deepspeech/predict.py
COPY deepspeech_pause_splitter /usr/bin/deepspeech_pause_splitter
COPY deepspeech_predict /usr/bin/deepspeech_predict
COPY deepspeech_train /usr/bin/deepspeech_train

# sym-link other utilities as "/usr/bin/deepspeech_"
RUN cd /DeepSpeech/bin/; \
    for i in *.py; do LINK="/usr/bin/deepspeech_`echo $i | sed s/\.py//g`"; ln -s /DeepSpeech/bin/$i $LINK; done;

ENV PYTHONPATH=/opt/deepspeech

WORKDIR /
