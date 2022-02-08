#!/bin/bash

model_name=${PWD##*/}

dvc run \
--name "fit_${model_name}" \
--deps ../../data/prepared \
--deps ../../data/morph \
--deps ../../data/handcrafted \
--outs ../../model_checkopoints/$model_name \
--outs-no-cache ../../oof_predictions/$model_name \
--plots-no-cache ../../plots/$model_name/prc_0.json \
--plots-no-cache ../../plots/$model_name/prc_1.json \
--plots-no-cache ../../plots/$model_name/prc_2.json \
--plots-no-cache ../../plots/$model_name/prc_3.json \
--plots-no-cache ../../plots/$model_name/prc_4.json \
--plots-no-cache ../../plots/$model_name/prc_5.json \
--plots-no-cache ../../plots/$model_name/prc_6.json \
--plots-no-cache ../../plots/$model_name/prc_7.json \
--plots-no-cache ../../plots/$model_name/prc_8.json \
--plots-no-cache ../../plots/$model_name/roc_0.json \
--plots-no-cache ../../plots/$model_name/roc_1.json \
--plots-no-cache ../../plots/$model_name/roc_2.json \
--plots-no-cache ../../plots/$model_name/roc_3.json \
--plots-no-cache ../../plots/$model_name/roc_4.json \
--plots-no-cache ../../plots/$model_name/roc_5.json \
--plots-no-cache ../../plots/$model_name/roc_6.json \
--plots-no-cache ../../plots/$model_name/roc_7.json \
--plots-no-cache ../../plots/$model_name/roc_8.json \
--metrics-no-cache ../../scores/"scores_${model_name}.json" \
python ../../models/$model_name/fit.py
