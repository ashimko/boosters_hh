from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from model_config import N_SPLITS, RANDOM_STATE


def get_model(random_state: int = 42) -> Tuple[Pipeline, bool]:

    base_model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=random_state)
    calibrated_model = CalibratedClassifierCV(base_model, cv=N_SPLITS, n_jobs=-1, ensemble=True)
    model = MultiOutputClassifier(estimator=calibrated_model, n_jobs=-1)
    
    params = {'estimator__base_estimator__C': np.random.uniform(0.1, 25, 10)} # 
    cv = MultilabelStratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    gscv = GridSearchCV(
        estimator=model, 
        param_grid=params,
        scoring='f1_samples',
        cv=cv,
        verbose=5,
        n_jobs=-1,
        refit=True)

    return model, gscv
