#!/bin/bash

set -e
CHANNELS="-c file:///opt/conda/envs/rapids/conda-bld -c rlratzel -c conda-forge"

UPLOAD_FILE=`conda build ${CHANNELS} ./conda --output`
UPLOAD_FILES=$(echo ${UPLOAD_FILE}|sed -e 's/\-py[0-9][0-9]/\-py36/')
UPLOAD_FILES="${UPLOAD_FILES} $(echo ${UPLOAD_FILE}|sed -e 's/\-py[0-9][0-9]/\-py37/')"

conda build ${CHANNELS} --variants="{python: [3.6, 3.7]}" ./conda
if [ "$1" = "--publish" ]; then
    anaconda upload ${UPLOAD_FILES}
fi
