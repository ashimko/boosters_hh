import torch
from transformers import AutoTokenizer, AutoModel

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from config import DATA_PATH, PREPARED_DATA_PATH, TEXT_COLS
from utils import create_folder
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

BATCH_SIZE = 32
EMB_DIM = 312
EMB_NAME = 'rubert-tiny'  # 'sbert_large_nlu_ru'

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def max_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    max_embeddings, _ = torch.max(token_embeddings * input_mask_expanded, 1)
    return max_embeddings



def get_batch_embedding(sentences: List, tokenizer, model) -> Tuple[np.ndarray]:   

    t = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings.cpu().detach().numpy()


def get_embedding(text: pd.Series, tokenizer, model) -> Tuple[np.ndarray]:
    n = len(text)
    embeddings = np.zeros(shape=(len(text), EMB_DIM), dtype=np.float32)

    for i in tqdm(range(0, n, BATCH_SIZE), total=n//BATCH_SIZE+1):
        embedding = get_batch_embedding(
            sentences=text.iloc[i:i+BATCH_SIZE].tolist(),
            tokenizer=tokenizer,
            model=model
        )

        embeddings[i:i+BATCH_SIZE] = embedding
    return embeddings


def save_embedding(embed, index, prefix, text_col):
    embed = pd.DataFrame(
        data=embed,
        index=index,
        columns=[f'{EMB_NAME}_{text_col}_{i}_mean' for i in range(EMB_DIM)]
    )
    path = os.path.join(DATA_PATH, EMB_NAME)
    create_folder(path)

    embed.to_pickle(os.path.join(path, f'{prefix}_{text_col}_{EMB_NAME}.pkl'), protocol=4)


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))

    train = pd.read_pickle(os.path.join(PREPARED_DATA_PATH, 'train.pkl'))
    test = pd.read_pickle(os.path.join(PREPARED_DATA_PATH, 'test.pkl'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(f"cointegrated/{EMB_NAME}")
    model = AutoModel.from_pretrained(f"cointegrated/{EMB_NAME}")
    model.to(device)
    

    for text_col in TEXT_COLS:
        embed = get_embedding(train[text_col], tokenizer, model)
        save_embedding(embed, train.index, 'train', text_col)

        embed = get_embedding(test[text_col], tokenizer, model)
        save_embedding(embed, test.index, 'test', text_col)
    

if __name__ == '__main__':
    main()
