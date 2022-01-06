from typing import Dict, List

from numpy import float32, int8, zeros_like
from pandas import DataFrame, Series
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def evaluate(model: Pipeline, X: DataFrame, y: DataFrame, n_splits: int, 
             random_state: int = 42, metrics: List = None) -> Dict:
    cv = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_results = cross_validate(
        estimator=model,
        X=X, y=y, scoring=metrics,
        cv=cv, n_jobs=-1, verbose=10,  error_score='raise',
        return_estimator=True, return_train_score=True
    )

    oof_pred_labels = zeros_like(y, dtype=int8)
    for fold, (_, test_idx) in enumerate(cv.split(X, y)):
        oof_pred_labels[test_idx] = cv_results['estimator'][fold].predict(X.iloc[test_idx])
    oof_pred_labels = DataFrame(data=oof_pred_labels, columns=y.columns, index=y.index)
    return cv_results, oof_pred_labels
