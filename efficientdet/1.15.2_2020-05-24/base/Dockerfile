ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}tensorflow/tensorflow:1.15.2-gpu-py3

RUN apt-get update && \
    apt-get install -y --no-install-recommends git-core wget libglib2.0-0 libsm6 libxrender-dev libxext6 && \
    rm -rf /var/lib/apt/lists/* && \
    pip install Cython && \
    pip install contextlib2 "Pillow>=6.0.0" pyyaml lxml matplotlib scikit-image wai.annotations==0.3.5 wai.tfutils==0.0.3 "opencv-python<=4.2" python-image-complete "six>=1.12.0" "tensorflow-addons>=0.9.1" && \
    pip uninstall -y pycocotools && \
    pip install "git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI" && \
    rm -Rf /root/.cache/pip


RUN cd /opt && \
    git clone https://github.com/google/automl && \
    cd /opt/automl && \
    git checkout 786f9f459e52d9d90650b1635f200ffaf21c6677

COPY efficientdet_* /usr/bin/

WORKDIR /opt/automl/efficientdet
