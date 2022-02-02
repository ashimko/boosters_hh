import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from config import DATA_PATH, PREPARED_DATA_PATH, TEXT_COLS
import tensorflow_text
import tensorflow_hub as hub
from tqdm import tqdm
from utils import create_folder


BATCH_SIZE = 1024
EMB_DIM = 512
EMB_NAME = 'universal_sentence_encoder_multilingual'


def get_batch_embedding(sentences: List, model) -> Tuple[np.ndarray]:   
    return model(sentences)


def get_embedding(text: pd.Series, model) -> Tuple[np.ndarray]:
    n = len(text)
    embeddings = np.zeros(shape=(len(text), EMB_DIM), dtype=np.float32)

    for i in tqdm(range(0, n, BATCH_SIZE), total=n//BATCH_SIZE+1):
        embedding = get_batch_embedding(
            sentences=text.iloc[i:i+BATCH_SIZE].tolist(),
            model=model
        )
        embeddings[i:i+BATCH_SIZE] = embedding
    return embeddings


def save_embedding(embeddings, index, prefix, text_col):
    embeddings = pd.DataFrame(
        data=embeddings,
        index=index,
        columns=[f'{EMB_NAME}_{text_col}_{i}' for i in range(EMB_DIM)]
    )
    path = os.path.join(DATA_PATH, EMB_NAME)
    create_folder(path)
    embeddings.to_pickle(os.path.join(path, f'{prefix}_{text_col}_embed.pkl'))


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))

    train = pd.read_pickle(os.path.join(PREPARED_DATA_PATH, 'train.pkl'))
    test = pd.read_pickle(os.path.join(PREPARED_DATA_PATH, 'test.pkl'))

    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

    for text_col in TEXT_COLS:
        save_embedding(
            embeddings=get_embedding(train[text_col], model), 
            index=train.index, 
            prefix='train', 
            text_col=text_col)
        save_embedding(
            embeddings=get_embedding(test[text_col], model), 
            index=test.index, 
            prefix='test',
            text_col=text_col)
    

if __name__ == '__main__':
    main()
