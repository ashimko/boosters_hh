import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from config import PREPARED_DATA_PATH, MORPH_DATA_PATH, HANDCRAFTED_DATA_PATH
from utils import squeeze_pred_proba
from evaluate import get_one_opt_treshold, get_pred_labels, get_cv_results
from helper import load_catboost_model, save_metrics, save_predictions, save_metric_plots, save_catboost_model, save_treshold

from model import get_model
from model_config import MODEL_NAME, N_SPLITS, RANDOM_STATE


def fit():
    train = pd.concat([
        pd.read_pickle(os.path.join(PREPARED_DATA_PATH, 'train.pkl')),
        pd.read_pickle(os.path.join(MORPH_DATA_PATH, 'train.pkl')),
        pd.read_pickle(os.path.join(HANDCRAFTED_DATA_PATH, 'train.pkl')),
    ], axis=1)

    target = pd.read_pickle(os.path.join(PREPARED_DATA_PATH, 'target.pkl'))
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof_pred_proba = pd.DataFrame(
        data=np.zeros(shape=target.shape),
        columns=target.columns,
        index=target.index
    )
    oof_pred_labels = oof_pred_proba.copy().astype(np.int8)
    
    for target_col in target.columns:
        print(f'train on target column {target_col}...')
        for fold, (train_idx, val_idx) in enumerate(cv.split(X=train, y=target[target_col])):
            print(f'start training {MODEL_NAME}, fold {fold}...')
            model = get_model()
            X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
            y_train, y_val = target.iloc[train_idx, int(target_col)], target.iloc[val_idx, int(target_col)]

            # model.fit(
            #     X=X_train, 
            #     y=y_train,
            #     verbose=20,
            #     eval_set=(X_val, y_val),
            #     early_stopping_rounds=600
            # )
            # save_catboost_model(model, MODEL_NAME, target_col, fold)
            model = load_catboost_model(get_model(), MODEL_NAME, target_col, fold)

            val_pred_proba = squeeze_pred_proba(model.predict_proba(X_val))
            oof_pred_proba.iloc[val_idx, int(target_col)] = val_pred_proba

    print('getting best treshold...')
    opt_treshold = get_one_opt_treshold(target, oof_pred_proba)
    save_treshold(opt_treshold)
    val_pred_labels = np.where(oof_pred_proba.values >= opt_treshold, 1, 0)
    oof_pred_labels.loc[:, :] = val_pred_labels
            
    score = f1_score(target, oof_pred_labels, average='samples', zero_division=0)
    print(f'model name {MODEL_NAME}, f1_score: {score}')
        

    save_predictions(oof_pred_proba, 'oof', MODEL_NAME, 'pred_proba')
    save_predictions(oof_pred_labels, 'oof', MODEL_NAME, 'pred_labels')
    
    cv_results = get_cv_results(target, oof_pred_labels, oof_pred_proba)
    save_metrics(cv_results, MODEL_NAME)
    save_metric_plots(target, oof_pred_proba, MODEL_NAME)


if __name__ == '__main__':
    fit()
