from typing import Tuple
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier

from config import ORDERED_CATEGORIES, UNORDERED_CATEGORIES, TEXT_COLS


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
        n_estimators=500,
        cat_features=ORDERED_CATEGORIES+UNORDERED_CATEGORIES,
        text_features=TEXT_COLS,
        random_state=random_state,
        max_depth=7,
        text_processing=text_processing_options,
        task_type="GPU",
        devices='0:1',
        silent=True)
    model = MultiOutputClassifier(estimator=base_estimator, n_jobs=1)

    return model, hasattr(base_estimator, 'predict_proba')
