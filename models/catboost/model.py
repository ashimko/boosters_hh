from typing import Tuple

from catboost import CatBoostClassifier
from config import (MORPH_TAG_COLS, NORMALIZED_TEXT_COLS, ORDERED_CATEGORIES,
                    TEXT_COLS, UNORDERED_CATEGORIES)
from sklearn.pipeline import Pipeline


def get_model(random_state: int = 42) -> Tuple[Pipeline, bool]:
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

    model = CatBoostClassifier(
        n_estimators=1000,
        cat_features=ORDERED_CATEGORIES+UNORDERED_CATEGORIES,
        text_features=TEXT_COLS+NORMALIZED_TEXT_COLS+MORPH_TAG_COLS,
        random_state=random_state,
        max_depth=7,
        text_processing=text_processing_options,
        eval_metric='F1',
        task_type="GPU",
        devices='0:1',
        silent=True)

    return model
