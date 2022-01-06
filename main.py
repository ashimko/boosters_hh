from pandas import DataFrame

from config import METRICS
from helper import (filter_metrics, get_test_data, get_train_data,
                    process_cv_results, save_fitted_model, save_metric_plots,
                    save_metrics, save_models, save_predicted_labels,
                    save_predicted_proba)
from model import make_model
from evaluate import evaluate


def main(n_splits: int = 5, random_state: int = 42):
    train, target, = get_train_data()
    model = make_model(random_state=random_state)

    cv_results, oof_pred_labels = evaluate(
        X=train, y=target, model=model, metrics=METRICS,
        random_state=random_state, n_splits=n_splits)

    save_predicted_labels(oof_pred_labels, mode='train')

    cv_results, estimators = process_cv_results(cv_results, METRICS)
    save_models(estimators)
    save_metrics(cv_results, METRICS)
    
    test = get_test_data()
    model.fit(X=train, y=target)
    save_fitted_model(model)

    test_pred_labels = DataFrame(data=model.predict(test), index=test.index, columns=target.columns)
    save_predicted_labels(test_pred_labels, mode='test')

if __name__ == '__main__':
    main()
