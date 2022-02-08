import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
from config import PREPARED_DATA_PATH, MORPH_DATA_PATH, HANDCRAFTED_DATA_PATH
from utils import squeeze_pred_proba
from evaluate import get_pred_labels
from helper import save_predictions, load_model
from model_config import MODEL_NAME, N_SPLITS


def predict():
    test = pd.concat([
        pd.read_pickle(os.path.join(PREPARED_DATA_PATH, 'test.pkl')),
        pd.read_pickle(os.path.join(MORPH_DATA_PATH, 'test.pkl')),
        pd.read_pickle(os.path.join(HANDCRAFTED_DATA_PATH, 'test.pkl')),
    ], axis=1)

    target_columns = pd.read_pickle(os.path.join(PREPARED_DATA_PATH, 'target.pkl')).columns
    test_pred_proba = pd.DataFrame(
        data=np.zeros(shape=(len(test), len(target_columns))),
        columns=target_columns,
        index=test.index   
    )
    test_pred_labels = test_pred_proba.copy()
    
    for fold in range(N_SPLITS):
        print(f'start predicting {MODEL_NAME}, fold {fold}...')
        
        model = load_model(MODEL_NAME, fold)
        test_pred_proba += squeeze_pred_proba(model.predict_proba(test)) 

    test_pred_proba /= N_SPLITS
    save_predictions(test_pred_proba, 'test', MODEL_NAME, 'pred_proba')
    
    test_pred_labels = get_pred_labels(test_pred_proba.values)
    test_pred_labels = pd.DataFrame(data=test_pred_labels, index=test.index, columns=target_columns)
    save_predictions(test_pred_labels, 'test', MODEL_NAME, 'pred_labels')


if __name__ == '__main__':
    predict()
