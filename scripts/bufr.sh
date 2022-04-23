#!/bin/bash

export CUDA_VISIBLE_DEVICES="5"

python bufr.py      --data-root ./datasets/ \
                    --output-dir ./ \
                    --alg-configs-dir ./configs/EMNIST-DA/ \
                    --data-config ./configs/EMNIST-DA/dataset.yml \
                    --seed 123 \
                    --deterministic \
                    --n-workers 4 \
                    --pin-mem