#!/bin/bash

export CUDA_VISIBLE_DEVICES="1"
export PYTHONPATH="/home/ian/meta-lstm/"

python analysis/feature_restorability.py  --data-root ./datasets/ \
                                          --output-dir ./ \
                                          --alg-config ./configs/EMNIST-DA/feature_restorability.yml \
                                          --data-config ./configs/EMNIST-DA/dataset.yml \
                                          --n-workers 4 \
                                          --pin-mem