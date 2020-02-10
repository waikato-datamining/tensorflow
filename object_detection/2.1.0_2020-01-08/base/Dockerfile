# Ubuntu 18.04, CUDA 10.0, Python 3.6
ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}tensorflow/tensorflow:2.1.0-gpu-py3

RUN apt-get update && \
	apt-get install -y --no-install-recommends git-core wget libglib2.0-0 libsm6 libxrender-dev libxext6 && \
	rm -rf /var/lib/apt/lists/* && \
	pip install Cython && \
	pip install contextlib2 Pillow lxml matplotlib scikit-image wai.annotations wai.tfutils opencv-python python-image-complete && \
	rm -Rf /root/.cache/pip

RUN cd /opt && \
	git clone https://github.com/waikato-datamining/tensorflow.git && \
	mv tensorflow/object_detection-2.1.0_2020-01-08/base/objdet_* /usr/bin/. && \
	cd tensorflow && \
	git clone https://github.com/tensorflow/models && \
	cd models/research && \
	git reset --hard d7a267397bf33a8b127004175ed77688fd37d0fb && \
	wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip && \
	unzip protobuf.zip && \
	rm protobuf.zip && \
	./bin/protoc object_detection/protos/*.proto --python_out=.

ENV PYTHONPATH=/opt/tensorflow/models/research:/opt/tensorflow/models/research/slim:/opt/tensorflow/models/research/object_detection
	
WORKDIR /opt/tensorflow/
