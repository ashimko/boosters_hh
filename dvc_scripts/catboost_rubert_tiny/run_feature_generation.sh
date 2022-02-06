#!/bin/bash

model_name=${PWD##*/}

dvc run \
--name "feature_generation_${model_name}" \
--deps ../../data/prepared \
python ../../features/rubert_tiny.py
