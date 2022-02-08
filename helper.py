import json
import os
from typing import Dict

import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from sklearn.metrics import precision_recall_curve, roc_curve
from tensorflow.keras import Model
from tensorflow.keras.layers import TextVectorization

from config import *
from utils import create_folder, load_pickle, save_to_pickle


def save_metrics(cv_results: Dict, model_name: str = None) -> None:
    file_name = f'scores_{model_name}.json' if model_name else 'scores.json'
    score_path = os.path.join(SCORES_PATH, file_name)
    with open(score_path, "w") as f:
        json.dump(cv_results, f, indent=4)


def save_metric_plots(true_labels: DataFrame, pred_proba: DataFrame, model_name: str) -> None:
    for col in true_labels.columns:
        precision, recall, prc_thresholds = precision_recall_curve(true_labels[col], pred_proba[col])
        fpr, tpr, roc_thresholds = roc_curve(true_labels[col], pred_proba[col])

        path = os.path.join(PLOTS_PATH, model_name)
        create_folder(path)
        with open(os.path.join(path, f'prc_{col}.json'), "w") as fd:
            json.dump(
                {
                    "prc": [
                        {"precision": float(p), "recall": float(r), "threshold": float(t)}
                        for p, r, t in zip(precision, recall, prc_thresholds)
                    ]
                },
                fd,
                indent=4,
            )
        with open(os.path.join(path, f'roc_{col}.json'), "w") as fd:
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
        pred_type: str = 'pred_proba') -> None:
    file_name = f'{model_name}_{pred_type}.csv'
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


def save_catboost_model(model, model_name: str, target_col: str=None, fold: int = -1):
    file_name = f'fold_{fold}_{model_name}.cbm'
    path = os.path.join(MODEL_PATH, model_name, f'col_{target_col}')
    create_folder(path)
    model.save_model(os.path.join(path, file_name), format='cbm')


def load_catboost_model(model, model_name: str, target_col: str=None, fold: int = -1):
    file_name = f'fold_{fold}_{model_name}.cbm'
    path = os.path.join(MODEL_PATH, model_name, f'col_{target_col}')
    return model.load_model(os.path.join(path, file_name), format='cbm')


def save_model_to_pickle(model, model_name: str, fold: int = -1) -> None:
    file_name = f'fold_{fold}_{model_name}.pkl'
    path = os.path.join(MODEL_PATH, model_name)
    create_folder(path)
    save_to_pickle(model, os.path.join(path, file_name))


def get_checkpoint_path(model_name: str, fold: int = -1) -> None:
    file_name = f'fold_{fold}_{model_name}'
    path = os.path.join(MODEL_PATH, model_name)
    create_folder(path)
    return os.path.join(path, file_name)


def load_model_from_pickle(model_name: str, fold: int = -1) -> None:
    file_name = f'fold_{fold}_{model_name}.pkl'
    path = os.path.join(MODEL_PATH, model_name)
    return load_pickle(os.path.join(path, file_name))


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


def save_treshold(treshold: float, path: str ='.') -> None:
    with open(os.path.join(path, 'opt_tresh.txt'), 'w') as f:
        f.write(str(treshold))


def load_treshold(path: str ='.') -> float:
    with open(os.path.join(path, 'opt_tresh.txt'), 'r') as f:
        treshold = float(f.read())
    return treshold