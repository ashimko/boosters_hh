from typing import DefaultDict, Dict, List, Union
from collections import defaultdict

from numpy import float32, int8, ndarray, zeros_like, zeros
from numpy.core.shape_base import hstack, vstack
from pandas import DataFrame, Series
from sklearn.model_selection import KFold, cross_validate, train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, log_loss, recall_score, precision_score
import pickle
from config import CITY, UNORDERED_CATEGORIES

from helper import save_model

def update_with_label_metrics(y_true: DataFrame, y_pred: DataFrame, 
                      cv_results: DefaultDict, mode: str = 'test') -> DataFrame:
    main_score = f1_score(y_true, y_pred, average='samples', zero_division=0)
    print('f1_score', main_score)
    cv_results[f'{mode}_f1_samples'].append(main_score)
    cv_results[f'{mode}_recall_samples'].append(recall_score(y_true, y_pred, average='samples', zero_division=0))
    cv_results[f'{mode}_precision_samples'].append(precision_score(y_true, y_pred, average='samples', zero_division=0))
    return cv_results

def update_with_pred_proba_metrics(y_true: DataFrame, y_pred: DataFrame, 
                      cv_results: DefaultDict, mode: str = 'test') -> DataFrame:
    cv_results[f'{mode}_neg_log_loss'].append(-log_loss(y_true, y_pred))
    cv_results[f'{mode}_roc_auc_ovo'].append(roc_auc_score(y_true, y_pred, average='samples', multi_class='ovo'))
    return cv_results

def _process_pred_proba(pred_proba: Union[ndarray, list]) -> ndarray:
    if isinstance(pred_proba, ndarray):
        return pred_proba
    elif isinstance(pred_proba, List):
        return vstack([label_prob[:, 1] for label_prob in pred_proba]).T
    else:
        raise NotImplementedError()


def evaluate(model: Pipeline, train: DataFrame, target: Series, test: DataFrame, 
             n_splits: int, random_state: int = 42, metrics: List = None, has_predict_proba: bool = True) -> Dict:
    cv = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_pred_labels = DataFrame(data=zeros_like(target, dtype=int8), index=target.index, columns=target.columns)
    oof_pred_proba = DataFrame(data=zeros_like(target, dtype=float32), index=target.index, columns=target.columns)
    
    test_shape = (len(test), target.shape[1])
    ens_test_pred_labels = DataFrame(data=zeros(shape=test_shape, dtype=int8), index=test.index, columns=target.columns)
    ens_test_pred_proba = DataFrame(data=zeros(shape=test_shape, dtype=float32), index=test.index, columns=target.columns)

    cv_results = defaultdict(list)
    fold = 0
    for train_idx, test_idx in cv.split(X=train, y=target):
        print('fold', fold)

        model.fit(train.iloc[train_idx], target.iloc[train_idx])
        save_model(model, fold)

        oof_pred_labels.iloc[test_idx] = model.predict(train.iloc[test_idx])
        cv_results = update_with_label_metrics(y_true=target.iloc[test_idx], y_pred=oof_pred_labels.iloc[test_idx],
                                               cv_results=cv_results, mode='test')
        cv_results = update_with_label_metrics(y_true=target.iloc[train_idx], y_pred=model.predict(train.iloc[train_idx]),
                                               cv_results=cv_results, mode='train')

        if has_predict_proba:
            oof_pred_proba.iloc[test_idx] = _process_pred_proba(model.predict_proba(train.iloc[test_idx]))
            cv_results = update_with_pred_proba_metrics(
                y_true=target.iloc[test_idx], 
                y_pred=oof_pred_labels.iloc[test_idx],
                cv_results=cv_results, mode='test')

            cv_results = update_with_pred_proba_metrics(
                y_true=target.iloc[train_idx], 
                y_pred=_process_pred_proba(model.predict_proba(train.iloc[train_idx])),
                cv_results=cv_results, mode='train')
        
        ens_test_pred_labels += model.predict(test)
        if has_predict_proba:
            ens_test_pred_proba += _process_pred_proba(model.predict_proba(test))
        fold += 1
    
    model.fit(train, target)
    save_model(model, fold=-1)
    pred = model.predict(test)
    ens_test_pred_labels += pred
    

    whole_train_test_pred_labels = DataFrame(data=pred, index=test.index, columns=target.columns)
    if has_predict_proba:
        pred_proba = _process_pred_proba(model.predict_proba(test))
        ens_test_pred_proba += pred_proba
        whole_train_test_pred_proba = DataFrame(data=pred_proba, index=test.index, columns=target.columns) 
        
    ens_test_pred_labels /= (n_splits + 1)
    ens_test_pred_proba /= (n_splits + 1)

    return (cv_results, oof_pred_labels, oof_pred_proba, 
            ens_test_pred_labels, ens_test_pred_proba, 
            whole_train_test_pred_labels, whole_train_test_pred_proba)
