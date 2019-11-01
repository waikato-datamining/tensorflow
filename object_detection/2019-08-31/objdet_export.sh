#!/bin/bash

cd /opt/tensorflow/models/research
python object_detection/export_inference_graph.py "$@"
cd /opt/tensorflow/object_detection/2019-08-31