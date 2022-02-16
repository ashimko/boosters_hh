import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
from config import LABSE_PATH, ORDERED_CATEGORIES, PREPARED_DATA_PATH, HANDCRAFTED_DATA_PATH
from utils import squeeze_pred_proba
from helper import _process_pred_labels, save_predictions, load_catboost_model, load_treshold
from model_config import MODEL_NAME, N_SPLITS
from model import get_model


def predict():
    test = pd.concat(
        ([pd.read_pickle(os.path.join(LABSE_PATH, path)) for path in os.listdir(LABSE_PATH) if path.startswith('test')] 
          + [pd.read_pickle(os.path.join(HANDCRAFTED_DATA_PATH, 'test.pkl')),
             pd.read_pickle(os.path.join(PREPARED_DATA_PATH, 'test.pkl'))[ORDERED_CATEGORIES].astype(np.int32)
             ]
        ),
        axis=1)

    target_columns = pd.read_pickle(os.path.join(PREPARED_DATA_PATH, 'target.pkl')).columns
    test_pred_proba = pd.DataFrame(
        data=np.zeros(shape=(len(test), len(target_columns))),
        columns=target_columns,
        index=test.index   
    )
    test_pred_labels = test_pred_proba.copy()
    
    for fold in range(N_SPLITS):
        print(f'start predicting {MODEL_NAME}, fold {fold}...')
        for target_col in target_columns:
            print(f'predicting target column {target_col}...')
            model = load_catboost_model(get_model(), MODEL_NAME, target_col, fold)
            test_pred_proba[target_col] += squeeze_pred_proba(model.predict_proba(test))

    test_pred_proba /= N_SPLITS
    save_predictions(test_pred_proba, 'test', MODEL_NAME, 'pred_proba')
    
    opt_treshold = load_treshold()
    test_pred_labels = np.where(test_pred_proba.values >= opt_treshold, 1, 0)
    test_pred_labels = pd.DataFrame(data=test_pred_labels, index=test.index, columns=target_columns)
    save_predictions(test_pred_labels, 'test', MODEL_NAME, 'pred_labels')

    submition = test_pred_proba.apply(_process_pred_labels, axis=1).rename('target')
    save_predictions(submition, 'submit', MODEL_NAME, 'submit')

if __name__ == '__main__':
    predict()
