#!/bin/bash

model_name=${PWD##*/}

dvc run \
--name "predict_${model_name}" \
--deps ../../model_checkopoints/$model_name \
--outs-no-cache ../../test_predictions/$model_name \
--force \
--always-changed \
python ../../models/$model_name/predict.py
