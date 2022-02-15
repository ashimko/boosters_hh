import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(Path(__file__).parents[1]))
import pandas as pd
from config import (HANDCRAFTED_DATA_PATH, MORPH_DATA_PATH, ORDERED_CATEGORIES,
                    PREPARED_DATA_PATH)
from helper import get_checkpoint_path
from sklearn.preprocessing import StandardScaler
from utils import save_to_pickle, load_pickle

from model_config import MODEL_NAME, TEXT_ENC_COLS


def get_train() -> pd.DataFrame:

    text_features = pd.concat([
            pd.read_pickle(os.path.join(PREPARED_DATA_PATH, 'train.pkl')),
            pd.read_pickle(os.path.join(MORPH_DATA_PATH, 'train.pkl')),
            ], axis=1)[TEXT_ENC_COLS]

    real_features = pd.concat([
        pd.read_pickle(os.path.join(PREPARED_DATA_PATH, 'train.pkl'))[ORDERED_CATEGORIES],
        pd.read_pickle(os.path.join(HANDCRAFTED_DATA_PATH, 'train.pkl'))
        ], axis=1)

    scaler = StandardScaler()
    real_features.loc[:, :] = scaler.fit_transform(real_features.values)
    scaler_path = get_checkpoint_path(MODEL_NAME, 'scaler')
    save_to_pickle(scaler, scaler_path)

    return pd.concat([text_features, real_features], axis=1)


def get_test() -> pd.DataFrame():
    text_features = pd.concat([
            pd.read_pickle(os.path.join(PREPARED_DATA_PATH, 'test.pkl')),
            pd.read_pickle(os.path.join(MORPH_DATA_PATH, 'test.pkl')),
            ], axis=1)[TEXT_ENC_COLS]

    real_features = pd.concat([
        pd.read_pickle(os.path.join(PREPARED_DATA_PATH, 'test.pkl'))[ORDERED_CATEGORIES],
        pd.read_pickle(os.path.join(HANDCRAFTED_DATA_PATH, 'test.pkl'))
        ], axis=1)

    scaler_path = get_checkpoint_path(MODEL_NAME, 'scaler')
    scaler = load_pickle(scaler_path)
    real_features.loc[:, :] = scaler.transform(real_features.values)
    
    return pd.concat([text_features, real_features], axis=1)
