#!/bin/bash

cd object_detection/2019-08-31
arguments=""
for par in "$@"
do
	arguments="${arguments} $par"
done
python predict.py ${arguments}
cd ../..