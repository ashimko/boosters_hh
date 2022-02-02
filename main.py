from config import METRICS, TEXT_COLS, UNORDERED_CATEGORIES
from helper import (get_encoders, get_test_data, get_train_data,
                    process_cv_results, save_metric_plots,
                    save_metrics, save_predicted_labels,
                    save_predictions)
from model import VOCAB_SIZE, make_model
from evaluate import evaluate


def main(n_splits: int = 3, random_state: int = 42):
    train, target, = get_train_data()
    test = get_test_data()
    encoders = get_encoders(train[TEXT_COLS+UNORDERED_CATEGORIES], VOCAB_SIZE)
    model = make_model(encoders)

    (cv_results, 
    oof_pred_labels, oof_pred_proba, test_pred_labels, test_pred_proba, 
    whole_train_pred_labels, whole_train_pred_proba) = evaluate(
        train=train, target=target, test=test, model=model,
        random_state=random_state, n_splits=n_splits)

    save_predicted_labels(oof_pred_labels, mode='train')
    save_predictions(oof_pred_proba, mode='train')

    save_predicted_labels(test_pred_labels, mode='test_avg_by_folds')
    save_predictions(test_pred_proba, mode='test_avg_by_folds')

    save_predicted_labels(whole_train_pred_labels, mode='test_whole_train')
    save_predictions(whole_train_pred_proba, mode='test_whole_train')

    save_metric_plots(true_labels=target, pred_proba=oof_pred_proba)

    cv_results = process_cv_results(cv_results, METRICS)
    save_metrics(cv_results, METRICS)


if __name__ == '__main__':
    main()
