#!/bin/bash
# export all environment variables from .env file
while read line; do 
    export "$line"; 
done < "pipeline/scripts/env/sampling.env"

echo "Pulling nescessary file/folder for sampling module"
# dvc pull sampling dataset/sampling/sampling_dataset/wiki_sampling -r sampling
# dvc pull sampling dataset/sampling/sampling_dataset/wiki_sampling
PYTHONPATH=./ python pipeline/inference/dataset_sampling.py \
    "$@"
