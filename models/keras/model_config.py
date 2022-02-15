import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(Path(__file__).parents[1]))
from config import (MORPH_TAG_COLS, NORMALIZED_TEXT_COLS, TEXT_COLS,
                    UNORDERED_CATEGORIES)

MODEL_NAME = os.path.basename(os.path.dirname(__file__))

VOCAB_SIZE = 15000
BATCH_SIZE = 128
N_TARGETS = 9
N_EPOCHS = 35
N_SPLITS = 3
RANDOM_STATE = 78

TEXT_ENC_COLS = TEXT_COLS + UNORDERED_CATEGORIES + NORMALIZED_TEXT_COLS