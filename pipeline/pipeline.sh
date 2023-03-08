#!/bin/bash
CURRENT_PATH=$PWD

ENV_PATH="$CURRENT_PATH/.env"
# export all environment variables from .env file
while read line; do 
    export "$line"; 
done < "$ENV_PATH"


# RUN PROCESS DATASET
PYTHONPATH=./ python pipeline/dataset_constructor/prepare_dataset.py

ALL_FILE=`ls -a $CURRENT_PATH`
if ! [[ "${ALL_FILE[*]}" == *".dvc"* ]]; then
    dvc init
    dvc remote add -d minhbtc_storage "$STORAGE"
fi

TRAING_DATASET_PATH="$CURRENT_PATH/dataset/training"
dvc add "$TRAING_DATASET_PATH"
git add "$CURRENT_PATH/dataset/training.dvc" "$CURRENT_PATH/.gitignore"
git commit -m "Update dataset"
dvc push

PYTHONPATH=./ python pipeline/trainer/mbart_trainer.py
