import os
MODEL_NAME = os.path.basename(os.path.dirname(__file__))
N_SPLITS = 3
RANDOM_STATE = 99
MODEL_NAMES = [
    'keras',  'sklearn', 'catboost', 'catboost_rubert_tiny',
    'catboost_handcrafted', 'catboost_labse_en_ru', 'catboost_sbert_large_nlu_ru']