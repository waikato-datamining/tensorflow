# Ubuntu 18.04, CUDA 10.0, Python 3.6
ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}tensorflow/tensorflow:1.14.0-gpu-py3

RUN apt-get update && \
	apt-get install -y --no-install-recommends git-core wget libglib2.0-0 libsm6 libxrender-dev libxext6 && \
	rm -rf /var/lib/apt/lists/* && \
	pip install Cython && \
	pip install contextlib2 Pillow lxml matplotlib scikit-image wai.annotations wai.tfutils opencv-python python-image-complete && \
	rm -Rf /root/.cache/pip

RUN cd /opt && \
	git clone https://github.com/waikato-datamining/tensorflow.git && \
	mv tensorflow/object_detection/1.14.0_2019-08-31_vgg16/base/objdet_* /usr/bin/. && \
	cd tensorflow && \
	git clone https://github.com/tensorflow/models && \
	cd models/research && \
	git reset --hard b9ef963d1e84da0bb9c0a6039457c39a699ea149 && \
	wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip && \
	unzip -q protobuf.zip && \
	rm protobuf.zip && \
	./bin/protoc object_detection/protos/*.proto --python_out=.

# VGG16 modifications
COPY faster_rcnn_vgg_16_fixed_feature_extractor.py /opt/tensorflow/models/research/object_detection/models
COPY model_builder.py /opt/tensorflow/models/research/object_detection/builders
COPY model_builder_test.py /opt/tensorflow/models/research/object_detection/builders
COPY keras_vgg16_to_tf.py /opt/tensorflow/models/research/object_detection

ENV PYTHONPATH=/opt/tensorflow/models/research:/opt/tensorflow/models/research/slim:/opt/tensorflow/models/research/object_detection
	
WORKDIR /opt/tensorflow/
