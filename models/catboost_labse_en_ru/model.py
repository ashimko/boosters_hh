from typing import Tuple

from catboost import CatBoostClassifier
from config import (MORPH_TAG_COLS, NORMALIZED_TEXT_COLS, ORDERED_CATEGORIES,
                    TEXT_COLS, UNORDERED_CATEGORIES)
from sklearn.pipeline import Pipeline


def get_model(n_estimators: int = 3000, random_state: int = 42) -> Tuple[Pipeline, bool]:
    

    model = CatBoostClassifier(
        n_estimators=n_estimators if n_estimators > 0 else 3000,
        random_state=random_state,
        # cat_features=ORDERED_CATEGORIES,
        max_depth=6,
        eval_metric='F1',
        task_type="GPU",
        devices='0:1',
        silent=True)

    return model
