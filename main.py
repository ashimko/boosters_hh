from pandas import DataFrame
from sklearn.pipeline import Pipeline

from config import METRICS
from helper import (filter_metrics, get_test_data, get_train_data,
                    process_cv_results, save_metric_plots,
                    save_metrics, save_predicted_labels,
                    save_predicted_proba)
from model import make_model
from evaluate import evaluate


def main(n_splits: int = 3, random_state: int = 42):
    train, target, = get_train_data()
    test = get_test_data()
    model, has_predict_proba = make_model(random_state=random_state)
    metrics = METRICS if has_predict_proba else filter_metrics(METRICS)

    (cv_results, 
    oof_pred_labels, oof_pred_proba, test_pred_labels, test_pred_proba, 
    whole_train_pred_labels, whole_train_pred_proba) = evaluate(
        train=train, target=target, test=test, model=model, metrics=metrics,
        random_state=random_state, n_splits=n_splits,
        has_predict_proba=has_predict_proba)

    save_predicted_labels(oof_pred_labels, mode='train')
    save_predicted_proba(oof_pred_proba if has_predict_proba else oof_pred_labels, mode='train')

    save_predicted_labels(test_pred_labels, mode='test_avg_by_folds')
    save_predicted_proba(test_pred_proba, mode='test_avg_by_folds')

    save_predicted_labels(whole_train_pred_labels, mode='test_whole_train')
    save_predicted_proba(whole_train_pred_proba, mode='test_whole_train')

    save_metric_plots(true_labels=target, pred_proba=oof_pred_proba)

    cv_results = process_cv_results(cv_results, metrics)
    save_metrics(cv_results, metrics)


if __name__ == '__main__':
    main()
