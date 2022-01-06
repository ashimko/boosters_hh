from typing import Tuple
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from numpy import bool_, int32, vectorize, random
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import NMF, TruncatedSVD, SparsePCA

from config import NEGATIVE, ORDERED_CATEGORIES, POSITIVE, TEXT_COLS, UNORDERED_CATEGORIES


def make_model(random_state: int = 42) -> Tuple[Pipeline, bool]:
    random.seed(random_state)
    n_splits=2
    cv = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    text_processing = Pipeline(memory='.cache', verbose=True, steps=[
        ('vectorize', FeatureUnion(n_jobs=-1, transformer_list=[
            ('vec_word', CountVectorizer(analyzer='word', dtype=int32, ngram_range=(1,3))),
            ('vec_char', CountVectorizer(analyzer='char_wb', dtype=int32, ngram_range=(1, 5))),
            ])),
        ('select_features', 'passthrough'),
        ('tfidf', 'passthrough'),
    ])
    features_generation = ColumnTransformer(n_jobs=-1, verbose=True, transformers=[
        ('positive_col', text_processing, POSITIVE),
        ('negative_col', text_processing, NEGATIVE),
        ('ordered_categories_as_id', 'passthrough', ORDERED_CATEGORIES),
        ('ordered_categories_ohe', OneHotEncoder(dtype=bool_, handle_unknown='ignore'), ORDERED_CATEGORIES),
        ('unordered_categories_ohe', OneHotEncoder(dtype=bool_, handle_unknown='ignore'), UNORDERED_CATEGORIES)
    ])
    base_estimator = MultiOutputClassifier(
        estimator=LinearSVC(random_state=random_state),
        n_jobs=-1)
    
    base_pipe = Pipeline(memory='.cache', verbose=True, steps=[
        ('get_features', features_generation),
        ('model', base_estimator)
    ])

    param_grid = {} # 'model__estimator__C': random.uniform(0.1, 10, 3)
    for text_col in TEXT_COLS:
        param_grid.update({f'get_features__{text_col.lower()}_col__select_features': [
                'passthrough', SelectPercentile(chi2, percentile=5), 
                SelectPercentile(chi2, percentile=25), SelectPercentile(chi2, percentile=50)],
            f'get_features__{text_col.lower()}_col__tfidf': ['passthrough', TfidfTransformer()]})

    model = GridSearchCV(
        estimator=base_pipe,
        param_grid=param_grid,
        cv=cv, verbose=3,
        scoring='f1_samples',
        n_jobs=-1
    )
    return model
