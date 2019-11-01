#!/bin/bash

cd /opt/tensorflow/tfrecords/adams2objectdetection
python convert.py "$@"
cd /opt/tensorflow/object_detection/2019-08-31