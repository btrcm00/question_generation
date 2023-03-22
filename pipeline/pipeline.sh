#!/bin/bash
CURRENT_PATH=$PWD

ENV_PATH="$CURRENT_PATH/.env"
# export all environment variables from .env file
while read line; do 
<<<<<<< HEAD
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
=======
    export "$line";
done < "$ENV_PATH"

PIPELINE_MODE="$1"
echo "PIPELINE MODE <<<<<< [$PIPELINE_MODE]"
shift
KWARGS="$@"
case "$PIPELINE_MODE" in 
    "train")
        echo "TRAINGING ..."
        sh ./pipeline/scripts/train.sh "$KWARGS"
    ;;
    "prepare_data")
        echo "PREPARING DATASET ..."
        sh ./pipeline/scripts/data_preparing.sh "$KWARGS"
    ;;
    "sampling")
        echo "SAMPLING DATASET ..."
        sh ./pipeline/scripts/sampling.sh "$KWARGS"
    ;;
    "all")
        echo "RUNNING ALL PIPELINE ... "
        sh ./pipeline/scripts/data_preparing.sh
        sh ./pipeline/scripts/train.sh
    ;;
    "api")
        echo "RUNNING API ... "
        sh ./pipeline/scripts/api.sh "$KWARGS"
    ;;
    *)
        echo "MISSING PIPELINE MODE"
        echo "It should be one of ['train', 'prepare_data', 'all']"
esac

# ALL_FILE=`ls -a $CURRENT_PATH`
# if ! [[ "${ALL_FILE[*]}" == *".dvc"* ]]; then
#     dvc ini
#     dvc remote add -d minhbtc_storage "$STORAGE"
# fi

# TRAING_DATASET_PATH="$CURRENT_PATH/dataset/training"
# dvc add "$TRAING_DATASET_PATH"
# git add "$CURRENT_PATH/dataset/training.dvc" "$CURRENT_PATH/.gitignore"
# git commit -m "Update dataset"
# dvc push
>>>>>>> af1f993 (refactor all pipeline and add script files for sub modules)
