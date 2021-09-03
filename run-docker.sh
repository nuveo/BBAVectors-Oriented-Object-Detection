#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters"
	exit
fi

weights_dir="${1}"
image_name=$(basename -- "$2")
image_dir="$( cd "$(dirname "$2")" >/dev/null 2>&1 ; pwd -P )"
resolution=$3
results_path=$(pwd)/results

mkdir -p $results_path

docker run --rm -it --gpus=all \
	-v "$image_dir":/image_dir/ \
	-v "$weights_dir":/weights_dir/ \
	-v "$results_path":/results/ \
	bbavectors \
	python ./bbavectors/app/object_detection.py \
		--model_dir /weights_dir/"$weight_dir" \
		--image /image_dir/"$image_name" \
		--resolution $resolution --plot

