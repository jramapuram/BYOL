#!/bin/bash

# first grab the root directory
ROOT_DIR=$(git rev-parse --show-toplevel)
echo "using root ${ROOT_DIR}"

# use the following command
CMD=$1
echo "executing: $CMD "

# run on the following GPU
GPU=${2:-0}
echo "using GPU: $GPU"

# execute it in docker
nvidia-docker run --ipc=host -v $HOME/datasets:/datasets -v $HOME/models:/models -v ${ROOT_DIR}:/workspace -e NVIDIA_VISIBLE_DEVICES=$GPU -it jramapuram/pytorch:1.5.0-cuda10.1 $CMD
