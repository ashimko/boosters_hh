from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.pipeline import Pipeline
from model_config import N_SPLITS, RANDOM_STATE


def get_model(random_state: int = 42) -> Tuple[Pipeline, bool]:

    base_model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=random_state)
    multiout_model = MultiOutputClassifier(estimator=base_model, n_jobs=-1)
    
    params = {'estimator__C': np.random.uniform(0.1, 25, 10)}
    cv = MultilabelStratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    model = GridSearchCV(
        estimator=multiout_model, 
        param_grid=params,
        scoring='f1_samples',
        cv=cv,
        verbose=5,
        n_jobs=-1,
        refit=True)

    return model
