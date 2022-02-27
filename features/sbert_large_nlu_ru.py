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

BATCH_SIZE = 256
EMB_DIM = 1024
EMB_NAME = 'sbert_large_nlu_ru'  # 'sbert_large_nlu_ru'

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



def get_batch_embedding(sentences: List, tokenizer, model, device) -> Tuple[np.ndarray]:   

    #Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=24, return_tensors='pt')
    encoded_input.to(device)

    #Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    #Perform pooling. In this case, mean pooling
    mean_pool = mean_pooling(model_output, encoded_input['attention_mask'])
    max_pool = max_pooling(model_output, encoded_input['attention_mask'])
    return mean_pool, max_pool


def get_embedding(text: pd.Series, tokenizer, model, device) -> Tuple[np.ndarray]:
    n = len(text)
    mean_pool_embed = np.zeros(shape=(len(text), EMB_DIM), dtype=np.float32)
    max_pool_embed = np.zeros(shape=(len(text), EMB_DIM), dtype=np.float32)

    for i in tqdm(range(0, n, BATCH_SIZE), total=n//BATCH_SIZE+1):
        mean_pool_sent, max_pool_sent = get_batch_embedding(
            sentences=text.iloc[i:i+BATCH_SIZE].tolist(),
            tokenizer=tokenizer,
            model=model,
            device=device
        )
        mean_pool_sent = mean_pool_sent.cpu().detach().numpy()
        max_pool_sent = max_pool_sent.cpu().detach().numpy()

        mean_pool_embed[i:i+BATCH_SIZE] = mean_pool_sent
        max_pool_embed[i:i+BATCH_SIZE] = max_pool_sent
    return mean_pool_embed, max_pool_embed


def save_embedding(mean_pool_embed, max_pool_embed, index, prefix, text_col):
    mean_pool_embed = pd.DataFrame(
        data=mean_pool_embed,
        index=index,
        columns=[f'{EMB_NAME}_{text_col}_{i}_mean' for i in range(EMB_DIM)]
    )
    max_pool_embed = pd.DataFrame(
        data=max_pool_embed,
        index=index,
        columns=[f'{EMB_NAME}_{text_col}_{i}_max' for i in range(EMB_DIM)]
    )
    path = os.path.join(DATA_PATH, EMB_NAME)
    create_folder(path)

    mean_pool_embed.to_pickle(os.path.join(path, f'{prefix}_{text_col}_mean_pool_embed.pkl'), protocol=4)
    max_pool_embed.to_pickle(os.path.join(path, f'{prefix}_{text_col}_max_pool_embed.pkl'), protocol=4)


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))

    train = pd.read_pickle(os.path.join(PREPARED_DATA_PATH, 'train.pkl'))
    test = pd.read_pickle(os.path.join(PREPARED_DATA_PATH, 'test.pkl'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(f"sberbank-ai/{EMB_NAME}")
    model = AutoModel.from_pretrained(f"sberbank-ai/{EMB_NAME}")
    model.to(device)
    

    for text_col in TEXT_COLS:
        mean_pool_embed, max_pool_embed = get_embedding(train[text_col], tokenizer, model, device)
        save_embedding(mean_pool_embed, max_pool_embed, train.index, 'train', text_col)

        mean_pool_embed, max_pool_embed = get_embedding(test[text_col], tokenizer, model, device)
        save_embedding(mean_pool_embed, max_pool_embed, test.index, 'test', text_col)
    

if __name__ == '__main__':
    main()
