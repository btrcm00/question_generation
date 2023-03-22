#!/bin/bash
# export all environment variables from .env file
while read line; do 
    export "$line"; 
done < "pipeline/scripts/env/sampling.env"

PYTHONPATH=./ python pipeline/inference/dataset_sampling.py \
    "$@"
