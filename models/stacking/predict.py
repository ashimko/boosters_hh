import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
from config import HANDCRAFTED_DATA_PATH, PREPARED_DATA_PATH
from utils import squeeze_pred_proba
from evaluate import get_pred_labels
from helper import _process_pred_labels, save_predictions, load_model_from_pickle, load_treshold
from model_config import MODEL_NAME, N_SPLITS, MODEL_NAMES


def predict():
    get_test_pred_proba_path = lambda model_name: f'../../test_predictions/{model_name}/{model_name}_pred_proba.csv'
    test = pd.concat([
        (pd.read_csv(get_test_pred_proba_path(model_name), index_col='review_id')
           .rename(columns={c: f'{model_name}_{c}' for c in map(str, range(9))}))
        for model_name in MODEL_NAMES], axis=1)

    # handcrafted = pd.read_pickle(os.path.join(HANDCRAFTED_DATA_PATH, 'test.pkl'))
    # test = test.join(handcrafted)
    
    target_columns = pd.read_pickle(os.path.join(PREPARED_DATA_PATH, 'target.pkl')).columns
    test_pred_proba = pd.DataFrame(
        data=np.zeros(shape=(len(test), len(target_columns))),
        columns=target_columns,
        index=test.index   
    )
    test_pred_labels = test_pred_proba.copy()
    
    for fold in range(N_SPLITS):
        print(f'start predicting {MODEL_NAME}, fold {fold}...')
        
        model = load_model_from_pickle(MODEL_NAME, fold)
        test_pred_proba += squeeze_pred_proba(model.predict_proba(test))
    
    model = load_model_from_pickle(MODEL_NAME, -1)
    test_pred_proba += squeeze_pred_proba(model.predict_proba(test))

    test_pred_proba /= (N_SPLITS + 1)
    save_predictions(test_pred_proba, 'test', MODEL_NAME, 'pred_proba')
    
    opt_treshold = load_treshold()
    test_pred_labels = np.where(test_pred_proba.values >= opt_treshold, 1, 0)
    test_pred_labels = pd.DataFrame(data=test_pred_labels, index=test.index, columns=target_columns)
    save_predictions(test_pred_labels, 'test', MODEL_NAME, 'pred_labels')

    submition = test_pred_labels.apply(_process_pred_labels, axis=1).rename('target')
    save_predictions(submition, 'submit', MODEL_NAME, 'submit')


if __name__ == '__main__':
    predict()
