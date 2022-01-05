from typing import DefaultDict, Dict, List, Union
from collections import defaultdict

from numpy import float32, int8, ndarray, zeros_like, zeros
from numpy.core.shape_base import hstack, vstack
from pandas import DataFrame, Series
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, log_loss, recall_score, precision_score

def update_with_label_metrics(y_true: DataFrame, y_pred: DataFrame, 
                      cv_results: DefaultDict, mode: str = 'test') -> DataFrame:
    cv_results[f'{mode}_f1_samples'].append(f1_score(y_true, y_pred, average='samples'))
    cv_results[f'{mode}_recall_samples'].append(recall_score(y_true, y_pred, average='samples'))
    cv_results[f'{mode}_precision_samples'].append(precision_score(y_true, y_pred, average='samples'))
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
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_pred_labels = DataFrame(data=zeros_like(target, dtype=int8), index=target.index, columns=target.columns)
    oof_pred_proba = DataFrame(data=zeros_like(target, dtype=float32), index=target.index, columns=target.columns)
    
    test_shape = (len(test), target.shape[1])
    test_pred_labels = DataFrame(data=zeros(shape=test_shape, dtype=int8), index=test.index, columns=target.columns)
    test_pred_proba = DataFrame(data=zeros(shape=test_shape, dtype=float32), index=test.index, columns=target.columns)

    cv_results = defaultdict(list)
    for train_idx, test_idx in cv.split(train):
        model.fit(train.iloc[train_idx], target.iloc[train_idx])
        cv_results['estimator'].append(model)

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
        
        test_pred_labels += model.predict(test)
        if has_predict_proba:
            test_pred_proba += _process_pred_proba(model.predict_proba(test))
    
    model.fit(train, target)
    test_pred_labels += model.predict(test)
    if has_predict_proba:
        test_pred_proba += _process_pred_proba(model.predict_proba(test))
        
    test_pred_labels /= (n_splits + 1)
    test_pred_proba /= (n_splits + 1)

    return cv_results, oof_pred_labels, oof_pred_proba, test_pred_labels, test_pred_proba
