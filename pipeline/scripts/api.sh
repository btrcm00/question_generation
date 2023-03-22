#!/bin/bash
# export all environment variables from .env file
while read line; do 
    export "$line"; 
done < "pipeline/scripts/env/api.env"

PYTHONPATH=./ python pipeline/inference/deploy.py \
    "$@"
