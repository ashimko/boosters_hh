import os
from typing import DefaultDict, Dict, List
from collections import defaultdict

import numpy as np
import tensorflow as tf
from pandas import DataFrame, Series
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, log_loss, recall_score, precision_score
from config import N_LABELS


def get_cv_results(
    y_true: DataFrame, pred_labels: DataFrame, pred_proba: DataFrame = None) -> Dict:
    cv_results = {}
    cv_results['f1_samples'] = f1_score(y_true, pred_labels, average='samples', zero_division=0)
    cv_results['recall_samples'] = recall_score(y_true, pred_labels, average='samples', zero_division=0)
    cv_results['precision_samples'] = precision_score(y_true, pred_labels, average='samples', zero_division=0)
    
    if pred_proba is not None:
        cv_results['log_loss'] = log_loss(y_true, pred_proba)
        cv_results['roc_auc_ovo'] = roc_auc_score(y_true, pred_proba, average='samples', multi_class='ovo')
    return cv_results


def expect_f1(y_prob, thres):
    idxs = np.where(y_prob >= thres)[0]
    tp = y_prob[idxs].sum()
    fp = len(idxs) - tp
    idxs = np.where(y_prob < thres)[0]
    fn = y_prob[idxs].sum()
    return 2*tp / (2*tp + fp + fn)


def optimal_threshold(y_prob):
    y_prob = np.sort(y_prob)[::-1]
    y_prob_unique = np.unique(y_prob)
    f1s = [expect_f1(y_prob, p) for p in y_prob_unique]
    thres = y_prob_unique[np.argmax(f1s)]
    return thres, f1s


def get_pred_labels(pred_proba: np.ndarray) -> np.ndarray:
    pred_labels = np.zeros_like(pred_proba, dtype=np.int8)
    for col_idx in range(N_LABELS):
        thres, _ = optimal_threshold(pred_proba[:, col_idx])
        pred_labels[:, col_idx] = np.where(pred_proba[:, col_idx] > thres, 1, 0)
    return pred_labels
