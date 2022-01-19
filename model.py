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

from config import NEGATIVE, ORDERED_CATEGORIES, POSITION_TXT, POSITIVE, UNORDERED_CATEGORIES, TEXT_COLS


def make_model(random_state: int = 42) -> Tuple[Pipeline, bool]:
    text_processing_options = {
        "tokenizers" : [{
            "tokenizer_id" : "Sense",
            "lowercasing" : "true",
            'separator_type': 'BySense'
        }],

        "dictionaries" : [{
            "dictionary_id" : "1-GramWord",
            "token_level_type": "Word",
            "gram_order" : "1"
        },{
            "dictionary_id" : "2-GramWord",
            "token_level_type": "Word",
            "gram_order" : "2"
        },{
            "dictionary_id" : "3-GramWord",
            "token_level_type": "Word",
            "gram_order" : "3"
        },{
            "dictionary_id" : "1-GramLetter",
            "token_level_type": "Letter",
            "gram_order" : "1"
        },{
            "dictionary_id" : "2-GramLetter",
            "token_level_type": "Letter",
            "gram_order" : "2"
        },{
            "dictionary_id" : "3-GramLetter",
            "token_level_type": "Letter",
            "gram_order" : "3"
        },{
            "dictionary_id" : "4-GramLetter",
            "token_level_type": "Letter",
            "gram_order" : "4"
        },{
            "dictionary_id" : "5-GramLetter",
            "token_level_type": "Letter",
            "gram_order" : "5"
        }],

        "feature_processing" : {
            "default" : [{
                "dictionaries_names" : [
                    "1-GramWord", "2-GramWord", "3-GramWord", 
                    "1-GramLetter", "2-GramLetter", "3-GramLetter", "4-GramLetter", "5-GramLetter"],
                "feature_calcers" : ["BoW", "NaiveBayes", "BM25"],
                "tokenizers_names" : ["Sense"]
            }]
        }
    }

    base_estimator = CatBoostClassifier(
        cat_features=ORDERED_CATEGORIES+UNORDERED_CATEGORIES,
        text_features=TEXT_COLS,
        random_state=random_state,
        max_depth=8,
        text_processing=text_processing_options,
        # task_type="GPU",
        # devices='0:1',
        verbose=10)
    model = MultiOutputClassifier(estimator=base_estimator, n_jobs=1)

    return model, hasattr(base_estimator, 'predict_proba')
