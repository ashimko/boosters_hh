#!/bin/bash

model_name=${PWD##*/}
for i in {0..8}; do
    dvc plots modify ../plots/$model_name/prc_$i.json --title "PR_RC_${i}" -x recall -y precision 
    dvc plots modify ../plots/$model_name/roc_$i.json --title "ROC_AUC_${i}" -x fpr -y tpr
done