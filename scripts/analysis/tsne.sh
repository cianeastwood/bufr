#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"
export PYTHONPATH="/home/ian/meta-lstm/"

python analysis/tsne.py