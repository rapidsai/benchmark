#!/bin/bash

set -e
CHANNELS="-c rlratzel -c conda-forge"

UPLOAD_FILE=`conda build ${CHANNELS} ./conda --output`
conda build ${CHANNELS} ./conda
if [ "$1" = "--publish" ]; then
    anaconda upload ${UPLOAD_FILE}
fi
