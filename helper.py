import json
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy import mean, ndarray
from pandas import DataFrame, Series, read_csv, read_pickle
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.pipeline import Pipeline
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras import Model
from utils import create_folder

from config import *
from utils import save_to_pickle


def save_metrics(cv_results: Dict, metrics) -> None:
    score_path = os.path.join(SCORES_PATH, 'scores.json')
    with open(score_path, "w") as f:
        json.dump(cv_results, f, indent=4)


def save_metric_plots(true_labels: DataFrame, pred_proba: DataFrame) -> None:
    for col in true_labels.columns:
        precision, recall, prc_thresholds = precision_recall_curve(true_labels[col], pred_proba[col])
        fpr, tpr, roc_thresholds = roc_curve(true_labels[col], pred_proba[col])
        
        nth_point = len(prc_thresholds) // 1000
        prc_points = list(zip(precision, recall, prc_thresholds))[::nth_point]

        with open(os.path.join(PLOTS_PATH, f'prc_{col}.json'), "w") as fd:
            json.dump(
                {
                    "prc": [
                        {"precision": float(p), "recall": float(r), "threshold": float(t)}
                        for p, r, t in prc_points
                    ]
                },
                fd,
                indent=4,
            )
        with open(os.path.join(PLOTS_PATH, f'roc_{col}.json'), "w") as fd:
            json.dump(
                {
                    "roc": [
                        {"fpr": float(fp), "tpr": float(tp), "threshold": float(t)}
                        for fp, tp, t in zip(fpr, tpr, roc_thresholds)
                    ]
                },
                fd,
                indent=4,
            )


def _process_pred_labels(x: Series) -> str:
        n = int((x >= 0.5).sum()) # if more than halve estimators predicted then predict at final
        if n == 0:
            return '0'
        return ','.join(map(str, sorted(x.nlargest(n).index)))


def save_predicted_labels(pred_labels: DataFrame, mode: str = 'test') -> None:
    pred_labels['target'] = pred_labels.apply(_process_pred_labels, axis=1)
    pred_labels = pred_labels['target']
    if mode == 'test_avg_by_folds':
        pred_labels.to_csv(os.path.join(SUBMITIONS_PATH, 'submit_labels.csv'))
    elif mode == 'test_whole_train':
        pred_labels.to_csv(os.path.join(SUBMITIONS_PATH, 'submit_labels_whole_train.csv'))
    elif mode == 'train':
        pred_labels.to_csv(os.path.join(OOF_PRED_PATH, 'submit_labels.csv'))
    
    else:
        raise NotImplementedError()


def save_predictions(
        predictions: DataFrame, 
        mode: str = 'test', 
        model_name: str = 'default',
        pred_type: str = 'pred_proba', 
        fold: int = -1) -> None:
    file_name = f'{model_name}_{pred_type}_{fold}.csv'
    test_path = os.path.join(TEST_PRED_PATH, model_name)
    create_folder(test_path)
    oof_path = os.path.join(OOF_PRED_PATH, model_name)
    create_folder(oof_path)

    if mode == 'test':
        predictions.to_csv(os.path.join(test_path, file_name))
    elif mode == 'oof':
        predictions.to_csv(os.path.join(oof_path, file_name))
    else:
        raise NotImplementedError()


def save_model(model: Model, model_name: str, fold: int = -1) -> None:
    file_name = f'fold_{fold}_{model_name}.pkl'
    path = os.path.join(MODEL_PATH, model_name)
    create_folder(path)
    save_to_pickle(model, os.path.join(path, file_name))


def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])


def get_encoders(data: DataFrame, vocab_size: int, **keywords) -> Dict:
    encoders = {}
    for col in data.columns:
        encoder = TextVectorization(max_tokens=vocab_size, name=col, **keywords)
        encoder.adapt(data[col])
        encoders[col] = encoder
    return encoders