#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0"

python pretrain.py  --data-root ./datasets/ \
                    --output-dir ./ \
                    --alg-config ./configs/EMNIST-DA/pretrain.yml \
                    --data-config ./configs/EMNIST-DA/dataset.yml \
                    --seed 123 \
                    --test-accuracy \
                    --deterministic \
                    --n-workers 4 \
                    --pin-mem

