from typing import Tuple
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from numpy import int32, int8, vectorize, bool_
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.decomposition import TruncatedSVD
import category_encoders as ce
from catboost import CatBoostClassifier

from config import NEGATIVE, ORDERED_CATEGORIES, POSITIVE, UNORDERED_CATEGORIES, TEXT_COLS


def make_model(n_splits: int = 5, random_state: int = 42) -> Tuple[Pipeline, bool]:
    text_processing_options = {
    "tokenizers" : [{
        "tokenizer_id" : "Space",
        "delimiter" : " ",
        "lowercasing" : "true"
    }],

    "dictionaries" : [{
        "dictionary_id" : "BiGram",
        "gram_order" : "2"
    }, {
        "dictionary_id" : "Word",
        "gram_order" : "1"
    }],

    "feature_processing" : {
        "default" : [{
            "dictionaries_names" : ["Word"],
            "feature_calcers" : ["BoW"],
            "tokenizers_names" : ["Space"]
        }],
        
        "1" : [{
            "tokenizers_names" : ["Space"],
            "dictionaries_names" : ["BiGram", "Word"],
            "feature_calcers" : ["BoW"]
        }, {
            "tokenizers_names" : ["Space"],
            "dictionaries_names" : ["Word"],
            "feature_calcers" : ["NaiveBayes"]
        }]
    }
}

    base_estimator = CatBoostClassifier(
        cat_features=ORDERED_CATEGORIES+UNORDERED_CATEGORIES,
        text_features=TEXT_COLS,
        random_state=random_state,
        text_processing=text_processing_options,
        verbose=10)
    model = MultiOutputClassifier(estimator=base_estimator, n_jobs=1)
    return model, hasattr(base_estimator, 'predict_proba')
