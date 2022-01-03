from typing import Dict, List

from pandas import DataFrame, Series
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

from config import METRICS
from helper import (filter_metrics, get_test_data, get_train_data, process_cv_results, save_fitted_model, save_metric_plots,
                    save_metrics, save_models, save_predicted_labels,
                    save_predicted_proba)
from model import make_model
from utils import save_to_pickle


def evaluate(model: Pipeline, X: DataFrame, y: Series, n_splits: int, random_state: int = 42, metrics: List = None) -> Dict:
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_results = cross_validate(
        estimator=model,
        X=X, y=y, scoring=metrics,
        cv = cv, n_jobs=-1, verbose=10,  error_score='raise',
        return_estimator=True, return_train_score=True
    ) 

    return cv_results


def main(n_splits: int = 5, random_state: int = 42):
    train, target, = get_train_data()
    model, has_predict_proba = make_model(n_splits=n_splits, random_state=random_state)
    metrics = METRICS if has_predict_proba else filter_metrics(METRICS)

    cv_results = evaluate(
        X=train, y=target, model=model, metrics=metrics,
        random_state=random_state, n_splits=n_splits)
    
    cv_results, estimators = process_cv_results(cv_results)
    save_models(estimators)
    save_metrics(cv_results, metrics)

    
    test = get_test_data()
    model.fit(X=train, y=target)
    save_fitted_model(model)

    pred_labels = DataFrame(data=model.predict(test), index=test.index, columns=target.columns)
    save_predicted_labels(pred_labels)

    pred_proba = DataFrame(data=model.predict_proba(test), index=test.index, columns=target.columns)
    save_predicted_proba(pred_proba)
    
    # TODO: rewrite cross_validate with cross_val_predict
    # save_metric_plots(target, pred_proba)


if __name__ == '__main__':
    main()
