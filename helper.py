import json
import os
from typing import Any, Dict, List, Tuple

from numpy import mean, ndarray
from pandas import DataFrame, Series, read_csv, read_pickle
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.pipeline import Pipeline

from config import (MODEL_PATH, OOF_PATH, PLOTS_PATH, PREPARED_DATA_PATH,
                    SCORES_PATH, SUBMITIONS_PATH)
from utils import save_to_pickle


def process_cv_results(cv_results: Dict, metrics: List) -> Tuple[Dict, List]:
    for metric in metrics:
        cv_results[f'avg_{metric}'] = mean(cv_results[f'test_{metric}'])
    return cv_results


def filter_metrics(metrics: List) -> List:
    return [metric for metric in metrics if metric not in ('roc_auc_ovo', 'neg_log_loss')]


def get_train_data() -> Tuple[DataFrame, Series]:
    X = read_pickle(os.path.join(PREPARED_DATA_PATH, 'train.pkl'))
    y = read_pickle(os.path.join(PREPARED_DATA_PATH, 'target.pkl'))
    return X, y

def get_test_data() -> DataFrame:
    return read_pickle(os.path.join(PREPARED_DATA_PATH, 'test.pkl'))


def save_metrics(cv_results: Dict, metrics) -> None:
    scores = {}
    for metric in metrics:
        avg_metric = f'avg_{metric}'
        scores[avg_metric] = cv_results.pop(avg_metric, None)
        
    score_path = os.path.join(SCORES_PATH, 'scores.json')
    with open(score_path, "w") as f:
        json.dump(scores, f, indent=4)

    cv_results_path = os.path.join(SCORES_PATH, 'cv_results.csv')
    DataFrame(cv_results).to_csv(cv_results_path, index=False)


def save_metric_plots(true_labels: DataFrame, pred_proba: DataFrame) -> None:
    for col in true_labels.columns:
        precision, recall, prc_thresholds = precision_recall_curve(true_labels[col], pred_proba[col])
        fpr, tpr, roc_thresholds = roc_curve(true_labels[col], pred_proba[col])


        with open(os.path.join(PLOTS_PATH, f'prc_{col}.json'), "w") as fd:
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
        pred_labels.to_csv(os.path.join(OOF_PATH, 'submit_labels.csv'))
    
    else:
        raise NotImplementedError()


def save_predicted_proba(pred_proba: DataFrame, mode: str = 'test') -> None:
    if mode == 'test_avg_by_folds':
        pred_proba.to_csv(os.path.join(SUBMITIONS_PATH, 'pred_proba.csv'))
    elif mode == 'test_whole_train':
        pred_proba.to_csv(os.path.join(SUBMITIONS_PATH, 'pred_proba_whole_train.csv'))
    elif mode == 'train':
        pred_proba.to_csv(os.path.join(OOF_PATH, 'pred_proba.csv'))
    else:
        raise NotImplementedError()


def save_model(estimator: Pipeline, fold: int = -1) -> None:
    if fold == -1:
        model_name = 'whole_train_target_model.pkl'
    else:
        model_name = f'fold_{fold}_target_model.pkl'
    save_to_pickle(obj=estimator, path=os.path.join(MODEL_PATH, model_name))
