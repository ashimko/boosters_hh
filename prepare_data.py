import os
import numpy as np
import pandas as pd
import tensorflow_hub as hub
import numpy as np
import tensorflow_text
from tqdm import tqdm

from config import (ID, ORDERED_CATEGORIES, ORIGINAL_DATA_PATH, POSITION,
                    PREPARED_DATA_PATH, TARGET, UNORDERED_CATEGORIES, TEXT_COLS)


def prepare_data(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    train=train.set_index(ID)
    train_idx = train.index

    test=test.set_index(ID)
    test_idx = test.index

    y = train[TARGET]
    y = y.str.get_dummies(sep=',').astype(np.int8)

    X = pd.concat((train.drop(TARGET, axis=1), test))
    for ordered_category in ORDERED_CATEGORIES:
        X[ordered_category] = X[ordered_category].fillna(-1)
        X[ordered_category] = X[ordered_category].astype('category').cat.as_ordered()
    
    X[UNORDERED_CATEGORIES] = X[UNORDERED_CATEGORIES].fillna('NA')
    X[UNORDERED_CATEGORIES] = X[UNORDERED_CATEGORIES].astype('category')
    X[TEXT_COLS] = X[TEXT_COLS].fillna('NA').astype('str')
    
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
    embeddings = []
    for text_col in TEXT_COLS:
        print(f'get embedding for {text_col}...')
        embedding = np.zeros(shape=(len(X), 512), dtype=np.float32)
        for i, sent in tqdm(enumerate(X[text_col]), total=len(X)):
            embedding[i] = embed([sent])[0]
        embedding = pd.DataFrame(
            data=embedding,
            index=X.index,
            columns=[f'{text_col}_embed_{i}' for i in range(512)])
        embeddings.append(embedding)
        print(embedding.memory_usage().sum() / 1e+6)
    X = X.join(embeddings)

    return X.loc[train_idx], X.loc[test_idx], y

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