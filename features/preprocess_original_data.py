import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(Path(__file__).parents[0]))

import os
from typing import List, Union

import numpy as np
import pandas as pd
import re
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
import tensorflow as tf

from config import (ID, ORDERED_CATEGORIES, ORIGINAL_DATA_PATH, PREPARED_DATA_PATH, TARGET, TEXT_COLS,
                    UNORDERED_CATEGORIES)


def _process_ordered_categories(
    data: DataFrame, ordered_categories: List, fillna_value: Union[int, None] =-1) -> DataFrame:

    for ordered_category in ordered_categories:
        data[ordered_category] = data[ordered_category].fillna(fillna_value)
        data[ordered_category] = data[ordered_category].astype('category').cat.as_ordered()
    return data

def _process_unorder_categories(
    data: DataFrame, unordered_categories: List,
    train_idx: Index, test_idx: Index, keep_only_common_categories: bool = True) -> DataFrame:
    for unordered_category in unordered_categories:
        data[unordered_category] = data[unordered_category].fillna('NA')

        if keep_only_common_categories:
            common_categories = (
                set(data.loc[train_idx, unordered_category]) 
                & set(data.loc[test_idx, unordered_category]))
            mask = ~data[unordered_category].isin(common_categories)
            data.loc[mask, unordered_category] = 'ANOTHER'
        data[unordered_category] = data[unordered_category].astype('category')
    return data


def _process_text_cols(data: DataFrame, text_cols: List, make_lower: bool = False) -> DataFrame:
    def __clean_numbers(x):
        if bool(re.search(r'\d', x)):
            x = re.sub('[0-9]{9,10}', '9201234567', x)
            x = re.sub('[0-9]{5,8}', '11111', x)
            x = re.sub('[0-9]{4}', '1111', x)
            x = re.sub('[0-9]{3}', '111', x)
            x = re.sub('[0-9]{2}', '11', x)
            x = re.sub('[0-9]{1}', '1', x)
        return x
    
    data[text_cols] = data[text_cols].fillna('NA').astype('str')

    for text_col in text_cols:
        data[text_col] = data[text_col].str.replace('\xa0', ' ')
        data[text_col] = data[text_col].str.replace('\ufeff', '')
        data[text_col] = data[text_col].str.replace('\u200d', ' ')
        data[text_col] = data[text_col].str.replace('ё', 'е')
        data[text_col] = data[text_col].str.replace('…', ' ... ')
        for quote in ["’", "‘", "´", "`"]:
            data[text_col] = data[text_col].str.replace(quote, "'")
        data[text_col] = data[text_col].apply(__clean_numbers)
        
    if make_lower:
        for col in text_cols:
            data[col] = data[col].str.lower()
    
    return data
        

def prepare_data(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    train=train.set_index(ID)
    train_idx = train.index

    test=test.set_index(ID)
    test_idx = test.index

    target = train[TARGET]
    target = target.str.get_dummies(sep=',').astype(np.int8)
    data = pd.concat((train.drop(TARGET, axis=1), test))

    data = _process_text_cols(data, TEXT_COLS, make_lower=False)
    data = _process_ordered_categories(data, ORDERED_CATEGORIES)
    data = _process_unorder_categories(
        data, UNORDERED_CATEGORIES, train_idx, test_idx, keep_only_common_categories=False
    )
    
    return data.loc[train_idx], data.loc[test_idx], target

def main() -> None:

    train, test, target = (
        prepare_data(
            train=pd.read_csv(os.path.join(ORIGINAL_DATA_PATH, 'train.csv')),
            test=pd.read_csv(os.path.join(ORIGINAL_DATA_PATH, 'test.csv'))
        )
    )

    train.to_pickle(os.path.join(PREPARED_DATA_PATH, 'train.pkl'), protocol=4)
    test.to_pickle(os.path.join(PREPARED_DATA_PATH, 'test.pkl'), protocol=4)
    target.to_pickle(os.path.join(PREPARED_DATA_PATH, 'target.pkl'), protocol=4)


if __name__ == '__main__':
    main()
