#!/bin/bash

model_name=${PWD##*/}

dvc run \
--name "preprocess_${model_name}" \
--deps ../../data/original \
--outs-persist ../../data/prepared \
python ../../features/preprocess_original_data.py
