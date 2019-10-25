#!/bin/bash

cd tfrecords/adams2objectdetection
arguments=""
for par in "$@"
do
	arguments="${arguments} $par"
done
python convert.py ${arguments}
cd ../..