#!/bin/bash

model_name=${PWD##*/}

dvc run \
--name "predict_${model_name}" \
--deps ../../data/prepared \
--deps ../../data/LaBSE-en-ru \
--deps ../../model_checkopoints/$model_name \
--outs-persist-no-cache ../../test_predictions/$model_name \
--force \
python ../../models/$model_name/predict.py
