import pickle
from typing import Any

def save_to_pickle(obj: Any, path:str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
