from typing import Tuple
from numpy import int32
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

from config import NEGATIVE, ORDERED_CATEGORIES, POSITIVE, UNORDERED_CATEGORIES


def make_model(n_splits: int = 5, random_state: int = 42) -> Tuple[Pipeline, bool]:
    features_generation = ColumnTransformer(n_jobs=-1, verbose=True, transformers=[
        ('positive_col', CountVectorizer(dtype=int32), POSITIVE),
        ('negative_col', CountVectorizer(dtype=int32), NEGATIVE),
        ('ordered_categories', 'passthrough', ORDERED_CATEGORIES),
        ('unordered_categories', OneHotEncoder(dtype=int32, handle_unknown='ignore'), UNORDERED_CATEGORIES)
    ])
    base_estimator = LinearSVC()
    model = Pipeline(memory='.cache', verbose=True, steps=[
        ('get_features', features_generation),
        ('model', MultiOutputClassifier(
            estimator=base_estimator,
            n_jobs=-1))
    ])
    return model, hasattr(base_estimator, 'predict_proba')
