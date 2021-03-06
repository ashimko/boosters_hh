import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
from config import PREPARED_DATA_PATH, TEXT_COLS, UNORDERED_CATEGORIES
from evaluate import get_pred_labels
from helper import _process_pred_labels, load_treshold, save_predictions, get_checkpoint_path
import tensorflow as tf
from model_config import MODEL_NAME, N_SPLITS
from model import get_model_input, get_model, get_encoders
from data_tools import get_train, get_test


def predict():
    train = get_train()
    test = get_test()

    target_columns = pd.read_pickle(os.path.join(PREPARED_DATA_PATH, 'target.pkl')).columns
    test_pred_proba = pd.DataFrame(
        data=np.zeros(shape=(len(test), len(target_columns))),
        columns=target_columns,
        index=test.index   
    )
    test_pred_labels = test_pred_proba.copy()
    
    for fold in range(N_SPLITS):
        print(f'start predicting {MODEL_NAME}, fold {fold}...')
        
        checkpoint_filepath = get_checkpoint_path(MODEL_NAME, fold)
        encoders = get_encoders(train[TEXT_COLS+UNORDERED_CATEGORIES])
        model = get_model(encoders)
        model.load_weights(checkpoint_filepath).expect_partial()
        test_pred_proba += model.predict(get_model_input(test))

    test_pred_proba /= N_SPLITS
    save_predictions(test_pred_proba, 'test', MODEL_NAME, 'pred_proba')
    
    opt_treshold = load_treshold()
    test_pred_labels = np.where(test_pred_proba.values >= opt_treshold, 1, 0)
    test_pred_labels = pd.DataFrame(data=test_pred_labels, index=test.index, columns=target_columns)
    save_predictions(test_pred_labels, 'test', MODEL_NAME, 'pred_labels')

    submition = test_pred_labels.apply(_process_pred_labels, axis=1).rename('target')
    save_predictions(submition, 'submit', MODEL_NAME, 'submit')


if __name__ == '__main__':
    predict()
