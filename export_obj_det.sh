#!/bin/bash

cd models/research
arguments=""
for par in "$@"
do
	arguments="${arguments} $par"
done
python object_detection/export_inference_graph.py ${arguments}
cd ../..