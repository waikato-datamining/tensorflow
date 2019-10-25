#!/bin/bash

cd models/research
arguments=""
for par in "$@"
do
	arguments="${arguments} $par"
done
python object_detection/model_main.py ${arguments}
cd ../..