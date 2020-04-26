#!/bin/sh

# pull the project
tar -C $HOME -xvf /tmp/project*.tar.gz

# pull some s3 dataset, be sure to set the correct S3 allowance for your nodes.
# aws s3 sync s3://bucket/dataset ~/datasets/mydataset

# Execute a simple VAE in the cloud for fashionMNIST
cd ~/ml_base && sh ./docker/run.sh "python vae_main.py --vae-type=simple --disable-gated --decoder-layer-type=conv --encoder-layer-type=conv --reparam-type=isotropic_gaussian --kl-beta=2 --task=fashion --batch-size=128 --visdom-url=http://neuralnetworkart.com --visdom-port=8097 --image-size-override=64 --uid=awsTest0_0"
