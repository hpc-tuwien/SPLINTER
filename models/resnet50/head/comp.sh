#!/bin/bash

for f in ./*.tflite
do
	echo "Compiling $f ..."
	edgetpu_compiler $f
done

