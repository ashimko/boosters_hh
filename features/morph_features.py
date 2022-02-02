import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pandas as pd
import pymorphy2
from  config import PREPARED_DATA_PATH, TEXT_COLS, MORPH_DATA_PATH


def lemmatize(string, morph):
    normal_forms = []
    tags = []
    for word in string.split():
        p = morph.parse(word)[0]
        normal_forms.append(p.normal_form)
        tag = []
        if p.tag.POS:
            tag.append(p.tag.POS) 
        if p.tag.animacy:
            tag.append(p.tag.animacy)
        if p.tag.aspect:
            tag.append(p.tag.aspect)
        if p.tag.case:
            tag.append(p.tag.case)
        if p.tag.gender:
            tag.append(p.tag.gender)
        if p.tag.involvement:
            tag.append(p.tag.involvement)
        if p.tag.mood:
            tag.append(p.tag.mood)
        if p.tag.number:
            tag.append(p.tag.number)
        if p.tag.person:
            tag.append(p.tag.person)
        if p.tag.tense:
            tag.append(p.tag.tense)
        if p.tag.transitivity:
            tag.append(p.tag.transitivity)
        if p.tag.voice:
            tag.append(p.tag.voice)
        tags.append(' '.join(tag))
    return pd.Series({
        'normalized': ' '.join(normal_forms),
        'tags': ', '.join(tags)
        })


def get_morph_features(text: pd.Series, morph) -> pd.DataFrame:
    morph_features = text.apply(lambda s: lemmatize(s, morph))
    morph_features = morph_features.rename(
        columns={'normalized': f'{text.name}_normalized',
                 'tags': f'{text.name}_tags'})
    return morph_features


def main():
    morph = pymorphy2.MorphAnalyzer()

    train = pd.read_pickle(os.path.join(PREPARED_DATA_PATH, 'train.pkl'))
    test = pd.read_pickle(os.path.join(PREPARED_DATA_PATH, 'test.pkl'))

    train_morph_features = pd.concat([get_morph_features(train[col], morph) for col in TEXT_COLS], axis=1)
    test_morph_features = pd.concat([get_morph_features(test[col], morph) for col in TEXT_COLS], axis=1)

    train_morph_features.to_pickle(os.path.join(MORPH_DATA_PATH, 'train.pkl'))
    test_morph_features.to_pickle(os.path.join(MORPH_DATA_PATH, 'test.pkl'))



if __name__ == '__main__':
    main()
