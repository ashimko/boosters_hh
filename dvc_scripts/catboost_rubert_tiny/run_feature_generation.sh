#!/bin/bash

model_name=${PWD##*/}

dvc run \
--name "feature_generation_${model_name}" \
--deps ../../data/prepared \
--outs-persist ../../data/rubert-tiny \
--force \
python ../../features/rubert_tiny.py
