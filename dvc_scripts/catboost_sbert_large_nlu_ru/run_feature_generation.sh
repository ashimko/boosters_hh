#!/bin/bash

model_name=${PWD##*/}

dvc run \
--name "feature_generation_${model_name}" \
--deps ../../data/prepared \
--outs-persist ../../data/morph \
--outs-persist ../../data/handcrafted \
python ../../features/sbert_large_nlu_ru.py
