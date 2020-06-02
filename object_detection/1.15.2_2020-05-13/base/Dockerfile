# Ubuntu 18.04, CUDA 10.0, Python 3.6
ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}tensorflow/tensorflow:1.15.2-gpu-py3

RUN apt-get update && \
	apt-get install -y --no-install-recommends git-core wget libglib2.0-0 libsm6 libxrender-dev libxext6 && \
	rm -rf /var/lib/apt/lists/* && \
	pip install Cython && \
	pip install contextlib2 Pillow lxml matplotlib scikit-image wai.annotations==0.3.5 wai.tfutils==0.0.3 "opencv-python<=4.2" python-image-complete && \
    pip uninstall -y numpy && \
    pip install "numpy<=1.17.4" && \
	rm -Rf /root/.cache/pip

RUN cd /opt && \
	git clone https://github.com/waikato-datamining/tensorflow.git && \
	cp tensorflow/object_detection/1.15.2_2020-05-13/base/objdet_* /usr/bin/. && \
	cd tensorflow && \
	git clone https://github.com/tensorflow/models && \
	cd models/research && \
	git reset --hard 8518d053936aaf30afb9ed0a4ea01baddca5bd17 && \
	wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip && \
	unzip -q protobuf.zip && \
	rm protobuf.zip && \
	./bin/protoc object_detection/protos/*.proto --python_out=.

ENV PYTHONPATH=/opt/tensorflow/models/research:/opt/tensorflow/models/research/slim:/opt/tensorflow/models/research/object_detection
	
WORKDIR /opt/tensorflow/
