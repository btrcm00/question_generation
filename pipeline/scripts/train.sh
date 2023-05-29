#!/bin/bash
# export all environment variables from .env file
while read line; do 
    export "$line"; 
done < "pipeline/scripts/env/train.env"

echo "Pulling dataset for training"
# dvc pull dataset/training -r training
dvc pull dataset/training

PYTHONPATH=./ python pipeline/trainer/mbart_trainer.py
