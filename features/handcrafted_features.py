import os
import sys
import string

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from typing import List

import pandas as pd
from collections import Counter
from config import HANDCRAFTED_DATA_PATH, PREPARED_DATA_PATH, TEXT_COLS


def _get_punct_count(data: pd.DataFrame, cols: List) -> pd.DataFrame:
    out = pd.DataFrame(index=data.index)
    for col in cols:
        for punct in '!*.,?@$-_—()/+£&':
            cnt_col = f'{col}_cnt_{punct}'
            out[cnt_col] = data[col].str.count(f'\{punct}')

            shr_col = f'{col}_shr_{punct}'
            out[shr_col] = out[cnt_col] / data[col].str.len()
    return out


def _get_len_review(data: pd.DataFrame, cols: List) -> pd.DataFrame:
    out = pd.DataFrame(index=data.index)
    for col in cols:
        col_name = f'len_review_{col}'
        out[col_name] = data[col].str.split().str.len()
    return out


def _get_num_unique_chars(data: pd.DataFrame, cols: List) -> pd.DataFrame:
    out = pd.DataFrame(index=data.index)
    for col in cols:
        col_name = f'cnt_unique_characters_{col}'
        out[col_name] = data[col].apply(lambda x: len(set(x)))
    return out


def _get_num_unique_words(data: pd.DataFrame, cols: List) -> pd.DataFrame:
    out = pd.DataFrame(index=data.index)
    for col in cols:
        cnt_col = f'cnt_unique_words_{col}'
        out[cnt_col] = data[col].apply(lambda comment: len(set(w for w in comment.split())))

        shr_col = f'shr_unique_words_{col}'
        out[shr_col] = out[cnt_col] / data[col].str.split().str.len()
    return out


def _get_num_repeated_words(data: pd.DataFrame, cols: List) -> pd.DataFrame:
    out = pd.DataFrame(index=data.index)
    for col in cols:
        cnt_col = f'cnt_repeated_words_{col}'
        out[cnt_col] = (
            data[col].str.replace(r'[^\w\s]+', '')
                     .apply(lambda t: sum([num for _, num in Counter(t.lower().split()).items() if num > 2])))

        shr_col = f'shr_repeated_words_{col}'
        out[shr_col] = out[cnt_col] / data[col].str.split().str.len()
    return out


def _get_share_english_letters(data: pd.DataFrame, cols: List) -> pd.DataFrame:
    out = pd.DataFrame(index=data.index)
    for col in cols:
        cnt_col = f'cnt_english_chars_{col}'
        out[cnt_col] = data[col].apply(lambda x: sum(char in string.ascii_letters for char in x))

        shr_col = f'shr_english_chars_{col}'
        out[shr_col] = out[cnt_col] / data[col].str.len()
    return out


def _get_share_upper_letters(data: pd.DataFrame, cols: List) -> pd.DataFrame:
    out = pd.DataFrame(index=data.index)
    for col in cols:
        cnt_col = f'cnt_upper_chars_{col}'
        out[cnt_col] = data[col].apply(lambda x: sum(1 for c in x if c.isupper()))

        shr_col = f'shr_upper_chars_{col}'
        out[shr_col] = out[cnt_col] / data[col].str.len()
    return out


def get_features(data: pd.DataFrame) -> pd.DataFrame:

    punctuation = _get_punct_count(data, TEXT_COLS)
    len_review = _get_len_review(data, TEXT_COLS)
    unique_chars = _get_num_unique_chars(data, TEXT_COLS)
    repeated_words = _get_num_repeated_words(data, TEXT_COLS)
    english_letters = _get_share_english_letters(data, TEXT_COLS)
    upper_letters = _get_share_upper_letters(data, TEXT_COLS)
    unique_words = _get_num_unique_words(data, TEXT_COLS)

    handcrafted = pd.concat([
        punctuation, len_review, unique_chars, repeated_words, 
        english_letters, upper_letters, unique_words], 
        axis=1)

    shr_cols = [col for col in handcrafted.columns if 'shr' in col]
    handcrafted['max_shr_val'] = handcrafted[shr_cols].max(axis=1)
    
    cnt_cols = [col for col in handcrafted.columns if 'cnt' in col]
    handcrafted['max_cnt_val'] = handcrafted[cnt_cols].max(axis=1)
    return handcrafted


def main():
    train = pd.read_pickle(os.path.join(PREPARED_DATA_PATH, 'train.pkl'))
    test = pd.read_pickle(os.path.join(PREPARED_DATA_PATH, 'test.pkl'))

    train_handcrafted = get_features(train)
    test_handcrafted = get_features(test)

    train_handcrafted.to_pickle(os.path.join(HANDCRAFTED_DATA_PATH, 'train.pkl'), protocol=4)
    test_handcrafted.to_pickle(os.path.join(HANDCRAFTED_DATA_PATH, 'test.pkl'), protocol=4)



if __name__ == '__main__':
    main()
