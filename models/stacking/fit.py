import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import f1_score
from config import HANDCRAFTED_DATA_PATH, PREPARED_DATA_PATH
from utils import squeeze_pred_proba
from evaluate import get_one_opt_treshold, get_cv_results
from helper import save_metric_plots, save_model_to_pickle, save_metrics, save_predictions, save_treshold

from model import get_model
from model_config import MODEL_NAME, N_SPLITS, RANDOM_STATE, MODEL_NAMES


def fit():
    get_oof_pred_proba_path = lambda model_name: f'../../oof_predictions/{model_name}/{model_name}_pred_proba.csv'
    train = pd.concat([
        (pd.read_csv(get_oof_pred_proba_path(model_name), index_col='review_id')
           .rename(columns={c: f'{model_name}_{c}' for c in map(str, range(9))}))
        for model_name in MODEL_NAMES], axis=1)
    # handcrafted = pd.read_pickle(os.path.join(HANDCRAFTED_DATA_PATH, 'train.pkl'))
    # train = train.join(handcrafted)

    target = pd.read_pickle(os.path.join(PREPARED_DATA_PATH, 'target.pkl'))

    oof_pred_proba = pd.DataFrame(
        data=np.zeros(shape=target.shape),
        columns=target.columns,
        index=target.index
    )
    oof_pred_labels = oof_pred_proba.copy()

    cv = MultilabelStratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    model, gscv = get_model()
    
    gscv.fit(X=train, y=target)
    save_model_to_pickle(gscv, MODEL_NAME, 'gscv')
    print('gscv best scores', gscv.best_score_)
    print('gscv best params', gscv.best_params_)
    model = gscv.best_estimator_

    for fold, (train_idx, val_idx) in enumerate(cv.split(X=train, y=target)):
        print(f'start training {MODEL_NAME}, fold {fold}...')
        
        X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
        y_train = target.iloc[train_idx]

        model.fit(X_train, y_train)
        save_model_to_pickle(model, MODEL_NAME, fold)

        val_pred_proba = squeeze_pred_proba(model.predict_proba(X_val))
        oof_pred_proba.iloc[val_idx] = val_pred_proba
    
    model.fit(train, target)
    save_model_to_pickle(model, MODEL_NAME, -1)
    
    print('getting best treshold...')
    opt_treshold = get_one_opt_treshold(target, oof_pred_proba)
    save_treshold(opt_treshold)
    oof_pred_labels.loc[:, :] = np.where(oof_pred_proba.values >= opt_treshold, 1, 0)

    save_predictions(oof_pred_proba, 'oof', MODEL_NAME, 'pred_proba')
    save_predictions(oof_pred_labels, 'oof', MODEL_NAME, 'pred_labels')
    
    cv_results = get_cv_results(target, oof_pred_labels, oof_pred_proba)
    save_metrics(cv_results, MODEL_NAME)
    save_metric_plots(target, oof_pred_proba, MODEL_NAME)


if __name__ == '__main__':
    fit()
