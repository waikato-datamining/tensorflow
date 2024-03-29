# Ubuntu 18.04, CUDA 10.0, Python 3.6
ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}tensorflow/tensorflow:1.14.0-gpu-py3

# update NVIDIA repo key
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
ARG distro=ubuntu1804
ARG arch=x86_64
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/3bf863cc.pub

RUN apt-get update && \
	apt-get install -y --no-install-recommends git-core wget libglib2.0-0 libsm6 libxrender-dev libxext6 && \
	rm -rf /var/lib/apt/lists/* && \
	pip install Cython && \
	pip install "opencv-python<4.2" && \
	pip install "Keras<2.3.0" && \
	pip install "keras-segmentation==0.3.0" && \
    pip uninstall -y numpy && \
    pip install "numpy<=1.17.4" && \
	pip install "python-image-complete" && \
	pip install "simple-file-poller>=0.0.9" && \
	rm -Rf /root/.cache/pip
RUN pip install --no-cache-dir redis "redis-docker-harness==0.0.1"

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
