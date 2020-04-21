# Ubuntu 18.04, CUDA 10.0, Python 3.6
ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}tensorflow/tensorflow:1.14.0-gpu-py3

RUN apt-get update && \
	apt-get install -y --no-install-recommends git-core wget libglib2.0-0 libsm6 libxrender-dev libxext6 && \
	rm -rf /var/lib/apt/lists/* && \
	pip install Cython && \
	pip install Keras==2.2.4 scikit-image && \
	rm -Rf /root/.cache/pip

RUN cd /opt && \
	git clone https://github.com/zhixuhao/unet.git && \
	cd unet && \
	git reset --hard b45af4d458437d8281cc218a07fd4380818ece4a

ENV PYTHONPATH=/opt/unet
	
WORKDIR /opt/unet/
