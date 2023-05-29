#!/bin/bash
# export all environment variables from .env file
while read line; do 
    export "$line"; 
done < "pipeline/scripts/env/data_preparing.env"

PYTHONPATH=./ python pipeline/dataset_constructor/prepare_dataset.py

TRAING_DATASET_PATH="$CURRENT_PATH/dataset/training"

echo "Pushing dataset and util files to dvc STORAGE"
dvc add "$TRAING_DATASET_PATH"
# git add "$CURRENT_PATH/dataset.dvc" "$CURRENT_PATH/.gitignore"
# git commit -m "Update dataset"
# dvc push "$TRAING_DATASET_PATH.dvc" -r training
# git commit -m "Update dataset"
dvc push "$TRAING_DATASET_PATH.dvc"

dvc add "$CURRENT_PATH/dataset/utils"
# dvc push dataset/utils.dvc -r utils
dvc push dataset/utils.dvc