import os

MODEL_NAME = os.path.basename(os.path.dirname(__file__))
TOKENIZER_NAME = 'bert-base-uncased'
MODEL_NAME = 'bert-base-cased'
MAX_SEQ_LEN = 669
RANDOM_STATE = 92
N_SPLITS = 3
BATCH_SIZE = 128
LR = 1e-6
EPOCHS = 5