from typing import Dict, List

from numpy import float32, int8, zeros_like
from pandas import DataFrame, Series
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline


def evaluate(model: Pipeline, X: DataFrame, y: Series, n_splits: int, 
             random_state: int = 42, metrics: List = None, has_predict_proba: bool = True) -> Dict:
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_results = cross_validate(
        estimator=model,
        X=X, y=y, scoring=metrics,
        cv=cv, n_jobs=-1, verbose=10,  error_score='raise',
        return_estimator=True, return_train_score=True
    )

    pred_labels = zeros_like(y, dtype=int8)
    pred_proba = zeros_like(y, dtype=float32)
    for fold, (_, test_idx) in enumerate(cv.split(X)):
        pred_labels[test_idx] = cv_results['estimator'][fold].predict(X.iloc[test_idx])
        if has_predict_proba:
            pred_proba[test_idx] = cv_results['estimator'][fold].predict_proba(X.iloc[test_idx])
    pred_labels = DataFrame(data=pred_labels, columns=y.columns, index=y.index)
    pred_proba = DataFrame(data=pred_proba, columns=y.columns, index=y.index)
    return cv_results, pred_labels, pred_proba
