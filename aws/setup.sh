#!/bin/sh

# pull docker container
nvidia-docker pull jramapuram/pytorch:1.1.0-cuda10.0

# make datasets dir in antipication of pulling datasets
mkdir ~/datasets

# install some deps [not super important, so dont break the build if it fails]
# do not un-comment; currently throws a DPKG error on the provided AMI =/
# sudo apt-get install -y emacs-nox htop tmux || true
