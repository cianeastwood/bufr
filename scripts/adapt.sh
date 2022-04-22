#!/bin/bash

export CUDA_VISIBLE_DEVICES="1"

python adapt.py     --data-root ./datasets/ \
                    --output-dir ./ \
                    --alg-configs-dir ./configs/EMNIST-DA/ \
                    --data-config ./configs/EMNIST-DA/dataset.yml \
                    --alg-name fr \
                    --seed 123 \
                    --deterministic \
                    --n-workers 4 \
                    --pin-mem
