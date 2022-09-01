# Ubuntu 18.04, CUDA 10.0, Python 3.6
ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}tensorflow/tensorflow:2.4.1-gpu

# update NVIDIA repo key
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
ARG distro=ubuntu1804
ARG arch=x86_64
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/3bf863cc.pub

RUN apt-get update && \
    apt-get install -y --no-install-recommends git-core wget libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install "Cython" && \
    pip install "numpy" && \
    pip install "h5py" && \
    pip install "tqdm" && \
    pip install "opencv-python" && \
    pip install "tqdm" && \
    pip install "imageio>=2.5.0" && \
    pip install "imgaug>=0.4.0" && \
    pip uninstall -y "keras" && \
    pip install "keras==2.4.3" && \
    pip install "python-image-complete" && \
    pip install "simple-file-poller>=0.0.9" && \
    pip install "redis" && \
    pip install "redis-docker-harness==0.0.1" && \
    pip install "simple-file-poller>=0.0.9" && \
    pip install "redis-docker-harness==0.0.1" && \
    rm -Rf /root/.cache/pip

RUN pip install --no-cache-dir "git+https://github.com/waikato-datamining/image-segmentation-keras.git"

COPY bash.bashrc /etc/bash.bashrc
COPY keras_seg /usr/bin/keras_seg
COPY keras_seg_conv /usr/bin/keras_seg_conv
COPY keras_seg_poll /usr/bin/keras_seg_poll
COPY keras_seg_redis /usr/bin/keras_seg_redis
COPY keras_seg_train /usr/bin/keras_seg_train
COPY convert.py /opt/image-segmentation-keras/convert.py
COPY predict.py /opt/image-segmentation-keras/predict.py
COPY predict_redis.py /opt/image-segmentation-keras/predict_redis.py
COPY train.py /opt/image-segmentation-keras/train.py

ENV PYTHONPATH=/opt/image-segmentation-keras

WORKDIR /
