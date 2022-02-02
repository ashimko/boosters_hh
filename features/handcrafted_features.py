import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from typing import List

import pandas as pd
from config import HANDCRAFTED_DATA_PATH, PREPARED_DATA_PATH, TEXT_COLS


def get_punct_count(data: pd.DataFrame, cols: List) -> pd.DataFrame:
    out = pd.DataFrame(index=data.index)
    for col in cols:
        for punct in '!*.,?@$-_()/':
            cnt_col = f'{col}_count_{punct}'
            out[cnt_col] = data[col].str.count(f'\{punct}')

            shr_col = f'{col}_shr_{punct}'
            out[shr_col] = out[cnt_col] / data[col].str.len()
    return out


def get_len_review(data: pd.DataFrame, cols: List) -> pd.DataFrame:
    out = pd.DataFrame(index=data.index)
    for col in cols:
        col_name = f'len_review_{col}'
        out[col_name] = data[col].str.split().str.len()
    return out


def main():
    train = pd.read_pickle(os.path.join(PREPARED_DATA_PATH, 'train.pkl'))
    test = pd.read_pickle(os.path.join(PREPARED_DATA_PATH, 'test.pkl'))

    train_punctuation_data = get_punct_count(train, TEXT_COLS)
    test_punctuation_data = get_punct_count(test, TEXT_COLS)

    train_len_review = get_len_review(train, TEXT_COLS)
    test_len_review = get_len_review(test, TEXT_COLS)

    train_handcrafted = pd.concat([train_punctuation_data, train_len_review], axis=1)
    test_handcrafted = pd.concat([test_punctuation_data, test_len_review], axis=1)

    train_handcrafted.to_pickle(os.path.join(HANDCRAFTED_DATA_PATH, 'train.pkl'))
    test_handcrafted.to_pickle(os.path.join(HANDCRAFTED_DATA_PATH, 'test.pkl'))



if __name__ == '__main__':
    main()
