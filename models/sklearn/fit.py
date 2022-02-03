import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import f1_score
from config import *
from utils import squeeze_pred_proba
from evaluate import get_pred_labels, get_cv_results
from helper import save_metric_plots, save_model, save_metrics, save_predictions

from model import get_model
from model_config import MODEL_NAME


def fit():
    train = pd.concat([
        pd.read_pickle(os.path.join(PREPARED_DATA_PATH, 'train.pkl')),
        pd.read_pickle(os.path.join(MORPH_DATA_PATH, 'train.pkl')),
        pd.read_pickle(os.path.join(HANDCRAFTED_DATA_PATH, 'train.pkl')),
    ], axis=1)

    target = pd.read_pickle(os.path.join(PREPARED_DATA_PATH, 'target.pkl'))
    cv = MultilabelStratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof_pred_proba = pd.DataFrame(
        data=np.zeros(shape=target.shape),
        columns=target.columns,
        index=target.index
    )
    oof_pred_labels = oof_pred_proba.copy()
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X=train, y=target)):
        print(f'start training {MODEL_NAME}, fold {fold}...')
        model = get_model()
        X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
        y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]

        model.fit(X_train, y_train)
        save_model(model, MODEL_NAME, fold)

        val_pred_proba = squeeze_pred_proba(model.predict_proba(X_val))
        oof_pred_proba.iloc[val_idx] = val_pred_proba

        val_pred_labels = get_pred_labels(val_pred_proba)
        oof_pred_labels.iloc[val_idx] = val_pred_labels
        
        score = f1_score(y_val, val_pred_labels, average='samples', zero_division=0)
        print(f'model name {MODEL_NAME}, fold {fold}, f1_score: {score}')
        

    save_predictions(oof_pred_proba, 'oof', MODEL_NAME, 'pred_proba')
    save_predictions(oof_pred_labels, 'oof', MODEL_NAME, 'pred_labels')
    
    cv_results = get_cv_results(target, oof_pred_labels, oof_pred_proba)
    save_metrics(cv_results, MODEL_NAME)
    save_metric_plots(target, oof_pred_proba, MODEL_NAME)


if __name__ == '__main__':
    fit()
