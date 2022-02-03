import os
import sys
from pathlib import Path
from typing import List

from catboost import CatBoostClassifier

sys.path.append(os.path.dirname(Path(__file__).parents[1]))

import category_encoders as ce
import numpy as np
import pandas as pd
from config import *

from sklearn.base import clone
from sklearn.compose import *
from sklearn.ensemble import *
from sklearn.feature_extraction.text import *
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import *
from sklearn.preprocessing import OneHotEncoder


def get_handcrafted_features_cols() -> List:
    return pd.read_pickle(os.path.join(HANDCRAFTED_DATA_PATH, 'train.pkl')).columns.tolist()


def get_model() -> Pipeline:
    base_model = LogisticRegression(solver='liblinear')

    char_features = ColumnTransformer(n_jobs=-1, transformers=[
        ('positive_char', TfidfVectorizer(analyzer='char', lowercase=False, dtype=np.float32), POSITIVE),
        ('negative_char', TfidfVectorizer(analyzer='char', lowercase=False, dtype=np.float32), NEGATIVE),
    ]
    )
    char_pipe = Pipeline(steps=[
        ('char_features', char_features),
        ('char_model', clone(base_model))
    ])

    word_features = ColumnTransformer(n_jobs=-1, transformers=[
        ('positive_word', TfidfVectorizer(analyzer='word', ngram_range=(1,3), dtype=np.float32), POSITIVE),
        ('negative_word', TfidfVectorizer(analyzer='word', ngram_range=(1,3), dtype=np.float32), NEGATIVE),
        ('position_word', TfidfVectorizer(analyzer='word', ngram_range=(1,3), dtype=np.float32), POSITION),
    ]
    )
    word_pipe = Pipeline(steps=[
        ('word_features', word_features),
        ('word_model', clone(base_model))
    ])

    word_features_normalized = ColumnTransformer(n_jobs=-1, transformers=[
        ('positive_normalized_word', CountVectorizer(analyzer='word', ngram_range=(1,3), dtype=np.float32), f'{POSITIVE}_normalized'),
        ('negative_normalized_word', CountVectorizer(analyzer='word', ngram_range=(1,3), dtype=np.float32), f'{NEGATIVE}_normalized'),
    ]
    )
    word_pipe_normalized = Pipeline(steps=[
        ('word_normalized_features', word_features_normalized),
        ('word_normalized_model', clone(base_model))
    ])

    tags_features = ColumnTransformer(n_jobs=-1, transformers=[
        ('positive_word', CountVectorizer(analyzer='word', ngram_range=(1,3), dtype=np.float32), f'{POSITIVE}_tags'),
        ('negative_word', CountVectorizer(analyzer='word', ngram_range=(1,3), dtype=np.float32), f'{NEGATIVE}_tags'),
    ]
    )
    tags_pipe = Pipeline(steps=[
        ('word_features', tags_features),
        ('word_model', clone(base_model))
    ])

    char_wb_features = ColumnTransformer(n_jobs=-1, transformers=[
        ('positive_char_wb', TfidfVectorizer(analyzer='char_wb', ngram_range=(1,3), dtype=np.float32), POSITIVE),
        ('negative_char_wb', TfidfVectorizer(analyzer='char_wb', ngram_range=(1,3), dtype=np.float32), NEGATIVE),
    ]
    )
    char_wb_pipe = Pipeline(steps=[
        ('char_wb_features', char_wb_features),
        ('char_wb_model', clone(base_model))
    ])

    unordered_cat_features = ColumnTransformer(n_jobs=-1, transformers=[
        ('unordered_cat', OneHotEncoder(dtype=np.float32, handle_unknown='ignore'), UNORDERED_CATEGORIES)
    ]
    )
    unordered_cat_pipe = Pipeline(steps=[
        ('unordered_cat_features', unordered_cat_features),
        ('unordered_cat_model', clone(base_model))
    ])

    ordered_cat_features = ColumnTransformer(n_jobs=-1, transformers=[
        ('ordered_cat', 'passthrough', ORDERED_CATEGORIES)
    ]
    )
    ordered_cat_pipe = Pipeline(steps=[
        ('ordered_cat_features', ordered_cat_features),
        ('ordered_cat_model', clone(base_model))
    ])

    cat_encoders_features = ColumnTransformer(n_jobs=-1, transformers=[
        ('WOEEncoder', ce.WOEEncoder(UNORDERED_CATEGORIES+ORDERED_CATEGORIES), UNORDERED_CATEGORIES+ORDERED_CATEGORIES),
        ('TargetEncoder', ce.TargetEncoder(UNORDERED_CATEGORIES+ORDERED_CATEGORIES), UNORDERED_CATEGORIES+ORDERED_CATEGORIES),
        ('CountEncoder', ce.CountEncoder(UNORDERED_CATEGORIES+ORDERED_CATEGORIES), UNORDERED_CATEGORIES+ORDERED_CATEGORIES),
        ('GLMMEncoder', ce.GLMMEncoder(ORDERED_CATEGORIES), ORDERED_CATEGORIES)
    ]
    )
    cat_encoders_pipe = Pipeline(steps=[
        ('unordered_cat_features', cat_encoders_features),
        ('unordered_cat_model', clone(base_model))
    ])

    manual_features = ColumnTransformer(n_jobs=-1, transformers=[
        ('manual_features', 'passthrough', get_handcrafted_features_cols())
    ]
    )
    manual_features_pipe = Pipeline(steps=[
        ('manual_features_features', manual_features),
        ('manual_features_model', CatBoostClassifier(silent=True, n_estimators=300))
    ])

    stacked_model = StackingClassifier(
        estimators=[
            ('char_pipe', char_pipe), 
            ('word_pipe', word_pipe),
            ('char_wb_pipe', char_wb_pipe),
            ('ordered_cat_pipe', ordered_cat_pipe),
            ('unordered_cat_pipe', unordered_cat_pipe),
            ('cat_encoders_pipe', cat_encoders_pipe),
            ('manual_features_pipe', manual_features_pipe),
            ('word_pipe_normalized', word_pipe_normalized),
            ('tags_pipe', tags_pipe)
        ],
        final_estimator=CatBoostClassifier(iterations=500, silent=True),
        cv=3,
        n_jobs=1
    )
    final_model = MultiOutputClassifier(estimator=stacked_model, n_jobs=-1)
    return final_model
