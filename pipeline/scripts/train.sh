#!/bin/bash
# export all environment variables from .env file
while read line; do 
    export "$line"; 
done < "pipeline/scripts/env/train.env"

PYTHONPATH=./ python pipeline/trainer/mbart_trainer.py
