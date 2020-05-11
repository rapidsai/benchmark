#!/bin/bash

set -e
CHANNELS="-c rlratzel -c conda-forge"

UPLOAD_FILE=`conda build ${CHANNELS} ./conda --output`
conda build ${CHANNELS} ./conda
anaconda upload ${UPLOAD_FILE}
