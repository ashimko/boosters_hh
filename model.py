import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict
from config import TEXT_COLS, ORDERED_CATEGORIES, UNORDERED_CATEGORIES


BUFFER_SIZE = 10000
VOCAB_SIZE = 10000
BATCH_SIZE = 64
N_TARGETS = 9
N_EPOCHS = 1


def get_model_input(data: pd.DataFrame) -> Dict:
    x = {f'{col}_text': data[col] for col in TEXT_COLS + UNORDERED_CATEGORIES}
    x.update({f'{col}_unordered_cat': data[col] for col in UNORDERED_CATEGORIES})
    x.update({'ordered_cat_input': data[ORDERED_CATEGORIES]})
    return x


def make_model(encoders: Dict) -> keras.Model:

    def _get_text_model(input, encoder):
        x = encoder(input)
        x = layers.Embedding(
                input_dim=len(encoder.get_vocabulary()),
                output_dim=64,
                mask_zero=True)(x)
        x = layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(N_TARGETS)(x)
        return x

    def _get_ordered_category_model(input):
        x = layers.Dense(32, activation='relu')(input)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(N_TARGETS)(x)
        return x

    def _get_unordered_category_mode(input, encoder):
        x = encoder(input)
        x = layers.CategoryEncoding(num_tokens=len(encoder.get_vocabulary()))(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(N_TARGETS)(x)
        return x

    def _get_final_classifier(features):
        x = layers.Dense(128, activation='relu')(features)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(N_TARGETS, activation='sigmoid')(x)
        return x

    text_inputs = {col: keras.Input(shape=(None,), dtype='string', name=f'{col}_text') for col in TEXT_COLS + UNORDERED_CATEGORIES}
    unordered_cat_inputs = {col: keras.Input(shape=(None,), dtype='string', name=f'{col}_unordered_cat') for col in UNORDERED_CATEGORIES}
    ordered_cat_input = keras.Input(shape=(len(ORDERED_CATEGORIES)), name='ordered_cat_input')

    ordered_cat_features = [_get_ordered_category_model(ordered_cat_input)]
    text_features = [_get_text_model(text_inputs[col], encoders[col]) for col in TEXT_COLS]
    unordered_cat_features = [_get_unordered_category_mode(unordered_cat_inputs[col], encoders[col]) for col in UNORDERED_CATEGORIES]

    features = layers.Add()(ordered_cat_features + text_features + unordered_cat_features)
    out = _get_final_classifier(features)

    model = keras.Model(
        inputs=[ordered_cat_input] + list(text_inputs.values()) + list(unordered_cat_inputs.values()),
        outputs=out
    )

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            tfa.metrics.F1Score(num_classes=9, average='micro', name='f1_score_micro'),
            tfa.metrics.F1Score(num_classes=9, average='macro', name='f1_score_macro')]
)

    return model
