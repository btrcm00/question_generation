#!/bin/bash
CURRENT_PATH=$PWD

ENV_PATH="$CURRENT_PATH/.env"
# export all environment variables from .env file
while read line; do 
    export "$line";
done < "$ENV_PATH"

# setup dvc
bash pipeline/scripts/dvc_setup.sh "$CURRENT_PATH"

PIPELINE_MODE="$1"
echo "PIPELINE MODE <<<<<< [$PIPELINE_MODE]"
shift
KWARGS="$@"
case "$PIPELINE_MODE" in
    "train")
        echo "TRAINING ..."
        bash pipeline/scripts/train.sh "$KWARGS"
    ;;
    "prepare_data")
        echo "PREPARING DATASET ..."
        bash pipeline/scripts/data_preparing.sh "$KWARGS"
    ;;
    "sampling")
        echo "SAMPLING DATASET ..."
        bash pipeline/scripts/sampling.sh "$KWARGS"
    ;;
    "all")
        echo "RUNNING ALL PIPELINE ... "
        bash pipeline/scripts/data_preparing.sh
        bash pipeline/scripts/train.sh
    ;;
    "api")
        echo "RUNNING API ... "
        bash pipeline/scripts/api.sh "$KWARGS"
    ;;
    *)
        echo "MISSING PIPELINE MODE"
        echo "It should be one of ['train', 'prepare_data', 'all']"
esac