import pickle
import os
from typing import Any, List
import numpy as np

def save_to_pickle(obj: Any, path:str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def create_folder(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def squeeze_pred_proba(pred_proba: List) -> np.ndarray:
    if not isinstance(pred_proba, List):
        return pred_proba
    return np.stack([p[:, 1] for p in pred_proba]).T
