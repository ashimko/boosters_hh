from typing import Tuple
from numpy import int32, vectorize
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

from config import NEGATIVE, ORDERED_CATEGORIES, POSITIVE, UNORDERED_CATEGORIES


def make_model(n_splits: int = 5, random_state: int = 42) -> Tuple[Pipeline, bool]:
    text_processing = Pipeline(memory='.cache', verbose=True, steps=[
        ('vectorize', CountVectorizer(dtype=int32)),
        ('select_features', SelectPercentile(chi2, percentile=25)),
        ('tfidf', TfidfTransformer())
    ])
    features_generation = ColumnTransformer(n_jobs=-1, verbose=True, transformers=[
        ('positive_col', text_processing, POSITIVE),
        ('negative_col', text_processing, NEGATIVE),
        ('ordered_categories', 'passthrough', ORDERED_CATEGORIES),
        ('unordered_categories', OneHotEncoder(dtype=int32, handle_unknown='ignore'), UNORDERED_CATEGORIES)
    ])
    base_estimator = SGDClassifier(random_state=random_state)
    model = Pipeline(memory='.cache', verbose=True, steps=[
        ('get_features', features_generation),
        ('model', MultiOutputClassifier(
            estimator=base_estimator,
            n_jobs=-1))
    ])
    return model, hasattr(base_estimator, 'predict_proba')
