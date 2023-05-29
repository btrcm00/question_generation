#!/bin/bash

# export all environment variables from .env file
while read line; do
    export "$line";
done < "pipeline/scripts/env/api.env"

# dvc pull pipeline/inference/checkpoint -r checkpoint
# dvc pull pipeline/inference/checkpoint
PYTHONPATH=./ python pipeline/inference/deploy.py \
    "$@"
