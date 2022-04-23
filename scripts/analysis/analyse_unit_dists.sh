#!/bin/bash

export CUDA_VISIBLE_DEVICES="1"
export PYTHONPATH="/home/ian/meta-lstm/"

python analysis/analyse_unit_dists.py   --data-root ./datasets/ \
                                        --output-dir ./ \
                                        --alg-config ./configs/EMNIST-DA/analyse_unit_dists.yml \
                                        --data-config ./configs/EMNIST-DA/dataset.yml \
                                        --n-workers 4 \
                                        --pin-mem
