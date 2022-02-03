#!/bin/bash

model_name=${PWD##*/}

dvc run \
--name "predict_${model_name}" \
--deps ../../data/prepared \
--deps ../../data/morph \
--deps ../../data/handcrafted \
--deps ../../model_checkpoints/$model_name
--outs-persist-no-cache ../../test_predictions/$model_name \
python ../../models/$model_name/predict.py
