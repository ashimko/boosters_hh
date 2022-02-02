import pickle
import os
from typing import Any

def save_to_pickle(obj: Any, path:str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def create_folder(path: str):
    if not os.path.exists(path):
        os.makedirs(path)