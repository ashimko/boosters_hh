from typing import Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from config import (MORPH_TAG_COLS, NORMALIZED_TEXT_COLS, ORDERED_CATEGORIES,
                    TEXT_COLS, UNORDERED_CATEGORIES)
from sklearn.pipeline import Pipeline


def get_model(random_state: int = 42) -> Tuple[Pipeline, bool]:

    base_model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=random_state)
    model = MultiOutputClassifier(estimator=base_model, n_jobs=-1) 

    return model
