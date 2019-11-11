# Ubuntu 18.04, CUDA 10.0, Python 3.6
FROM tensorflow/tensorflow:1.14.0-gpu-py3

RUN apt-get update && \
	apt-get install -y --no-install-recommends git-core wget && \
	rm -rf /var/lib/apt/lists/* && \
	pip install Cython contextlib2 Pillow lxml javaproperties matplotlib && \
	rm -Rf /root/.cache/pip

RUN cd /opt && \
	git clone https://github.com/waikato-datamining/tensorflow.git && \
	mv tensorflow/object_detection/2019-08-31/objdet_* /usr/bin/. && \
	cd tensorflow/tfrecords && \
	pip install . && \
	cd .. && \
	git clone https://github.com/tensorflow/models && \
	cd models/research && \
	git reset --hard b9ef963d1e84da0bb9c0a6039457c39a699ea149 && \
	wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip && \
	unzip protobuf.zip && \
	rm protobuf.zip && \
	./bin/protoc object_detection/protos/*.proto --python_out=. && \
	git clone https://github.com/cocodataset/cocoapi.git && \
	cd cocoapi/PythonAPI && \
	git reset --hard 636becdc73d54283b3aac6d4ec363cffbb6f9b20 && \
	make && \
	cp -r pycocotools /opt/tensorflow/models/research/
	
ENV PYTHONPATH=/opt/tensorflow/models/research:/opt/tensorflow/models/research/slim:/opt/tensorflow/models/research/object_detection:/opt/tensorflow/tfrecords/
	
WORKDIR /opt/tensorflow/
