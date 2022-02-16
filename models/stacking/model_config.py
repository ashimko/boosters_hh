import os
MODEL_NAME = os.path.basename(os.path.dirname(__file__))
N_SPLITS = 5
RANDOM_STATE = 79
MODEL_NAMES = ['keras', 'catboost_labse_en_ru', 'sklearn', 'catboost', 'catboost_rubert_tiny', 'catboost_sbert_large_nlu_ru', 'catboost_handcrafted']