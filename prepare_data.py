import os
from typing import List, Union

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
import tensorflow as tf

from config import (ID, ORDERED_CATEGORIES, ORIGINAL_DATA_PATH, POSITION,
                    POSITION_AS_TXT, PREPARED_DATA_PATH, TARGET, TEXT_COLS,
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


def _process_text_cols(data: DataFrame, text_cols: List, make_lower: bool = True) -> DataFrame:
    data[text_cols] = data[text_cols].fillna('NA').astype('str')
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
    data = _process_unorder_categories(data, UNORDERED_CATEGORIES, 
                                       train_idx, test_idx, keep_only_common_categories=False)
    
    return data.loc[train_idx], data.loc[test_idx], target

def main() -> None:

    train, test, target = (
        prepare_data(
            train=pd.read_csv(os.path.join(ORIGINAL_DATA_PATH, 'train.csv')),
            test=pd.read_csv(os.path.join(ORIGINAL_DATA_PATH, 'test.csv'))
        )
    )

    train.to_pickle(os.path.join(PREPARED_DATA_PATH, 'train.pkl'))
    test.to_pickle(os.path.join(PREPARED_DATA_PATH, 'test.pkl'))
    target.to_pickle(os.path.join(PREPARED_DATA_PATH, 'target.pkl'))


if __name__ == '__main__':
    main()
