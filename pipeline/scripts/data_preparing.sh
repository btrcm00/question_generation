#!/bin/bash
# export all environment variables from .env file
while read line; do 
    export "$line"; 
done < "pipeline/scripts/env/data_preparing.env"

PYTHONPATH=./ python pipeline/dataset_constructor/prepare_dataset.py
