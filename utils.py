import pickle
import os
from typing import Any, List
import numpy as np

def save_to_pickle(obj: Any, path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    with open(path, 'rb') as f:
        obj = pickle.load(path)
    return obj


def create_folder(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def squeeze_pred_proba(pred_proba: List) -> np.ndarray:
    if isinstance(pred_proba, List):
        return np.stack([p[:, 1] for p in pred_proba]).T
    if isinstance(pred_proba, np.ndarray) and len(pred_proba.shape) == 2:
        return pred_proba[:, 1]
    return pred_proba
