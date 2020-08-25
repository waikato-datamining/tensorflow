# Ubuntu 18.04, CUDA 10.0, Python 3.6
ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}tensorflow/tensorflow:1.14.0-gpu-py3

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
	rm -Rf /root/.cache/pip
	
COPY bash.bashrc /etc/bash.bashrc
COPY keras_seg /usr/bin/keras_seg
COPY keras_seg_poll /usr/bin/keras_seg_poll
COPY predict.py /opt/image-segmentation-keras

ENV PYTHONPATH=/opt/image-segmentation-keras

WORKDIR /
