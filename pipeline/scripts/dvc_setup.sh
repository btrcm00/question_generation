#!/bin/bash
echo "SETTING UP DVC ..."
# Set STORAGE environment variable corresponding to each server
if [[ "$HOSTNAME" == "ai24" ]]; then
    export STORAGE="/TMTAI/$STORAGE"
else
    export STORAGE="/AIHCM/$STORAGE"
fi

# current_dir="$1"
# Init dvc if not exist
# ALL_FILE=`ls -a $current_dir`
# if ! [[ "${ALL_FILE[*]}" == *".dvc"* ]]; then
#     git init
#     dvc init
# fi
git init
dvc init
# Add remote storage
dvc remote add -d qg_storage "$STORAGE" -f

# Config external cache
# dvc cache dir "$STORAGE/cache"
# dvc pull dataset/utils