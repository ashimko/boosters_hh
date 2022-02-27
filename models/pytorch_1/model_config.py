import os

MODEL_NAME = os.path.basename(os.path.dirname(__file__))
TOKENIZER_NAME = 'LaBSE-en-ru'
MODEL_NAME = 'LaBSE-en-ru'
MAX_SEQ_LEN = 669
RANDOM_STATE = 92
N_SPLITS = 3
LR = 1e-6
EPOCHS = 5
BATCH_SIZE = 32
EMB_DIM = 768