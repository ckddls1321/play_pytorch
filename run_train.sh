#!/bin/zsh
#
gradient jobs create \
    --name "cifar10" \
    --container "paperspace/fastai:1.0-CUDA9.2-base-3.0-v1.0.6" \
    --machineType "P5000" \
    --command "train_fastai.py ./configs/cifar10.py"
