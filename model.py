from typing import Tuple
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from numpy import int32, vectorize, bool_
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.decomposition import TruncatedSVD
import category_encoders as ce

from config import NEGATIVE, ORDERED_CATEGORIES, POSITIVE, UNORDERED_CATEGORIES


def make_model(n_splits: int = 5, random_state: int = 42) -> Tuple[Pipeline, bool]:
    text_processing = Pipeline(memory='.cache', verbose=True, steps=[
        ('vectorize', FeatureUnion(n_jobs=-1, transformer_list=[
            ('count_vec_char_wb', CountVectorizer(analyzer='char_wb', ngram_range=(1,4), dtype=int32)),
            ('count_vec_word', CountVectorizer(analyzer='word', ngram_range=(1,3), dtype=int32))])
            ),
        ('select_features', SelectPercentile(chi2, percentile=35)),
        ('tfidf', TfidfTransformer()),
    ])

    features_generation = ColumnTransformer(n_jobs=-1, verbose=True, transformers=[
        ('positive_col', text_processing, POSITIVE),
        ('negative_col', text_processing, NEGATIVE),
        ('ordered_categories_as_is', 'passthrough', ORDERED_CATEGORIES),
        ('ordered_categories_ohe', OneHotEncoder(dtype=bool_, handle_unknown='ignore'), ORDERED_CATEGORIES),
        ('unordered_categories', OneHotEncoder(dtype=bool_, handle_unknown='ignore'), UNORDERED_CATEGORIES),
    ])

    base_estimator = LinearSVC()
    model = Pipeline(memory='.cache', verbose=True, steps=[
        ('get_features', features_generation),
        ('model', MultiOutputClassifier(estimator=base_estimator, n_jobs=-1))
    ])
    return model, hasattr(base_estimator, 'predict_proba')
