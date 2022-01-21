import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_text as text
from typing import Dict
import tensorflow_hub as hub
from config import POSITION, TEXT_COLS, ORDERED_CATEGORIES, UNORDERED_CATEGORIES


VOCAB_SIZE = 15000
BATCH_SIZE = 4
N_TARGETS = 9
N_EPOCHS = 20


def get_model_input(data: pd.DataFrame) -> Dict:
    x = {f'{col}_text': data[col] for col in TEXT_COLS + UNORDERED_CATEGORIES}
    x.update({f'{col}_unordered_cat': data[col] for col in UNORDERED_CATEGORIES})
    x.update({'ordered_cat_input': data[ORDERED_CATEGORIES]})
    return x


def make_model(encoders: Dict) -> keras.Model:

    def _get_text_model(text_input):
        # preprocessor = hub.KerasLayer(
        #      "https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2")
        # encoder_inputs = preprocessor(text_input)
        encoder = hub.KerasLayer(
            "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3",
            trainable=False)
        outputs = encoder(text_input)
        # pooled_output = outputs["pooled_output"]
        # sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 768].
        x = layers.Dense(256)(outputs)
        x = keras.activations.relu(x, alpha=0.05)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(128)(x)
        x = keras.activations.relu(x, alpha=0.05)
        return x

    def _get_ordered_category_model(input):
        x = layers.Dense(128, activation='relu')(input)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128)(x)
        x = keras.activations.relu(x, alpha=0.05)
        return x

    def _get_unordered_category_mode(input, encoder):
        x = encoder(input)
        x = layers.CategoryEncoding(num_tokens=len(encoder.get_vocabulary()))(x)
        x = layers.Dense(128)(x)
        x = keras.activations.relu(x, alpha=0.05)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128)(x)
        x = keras.activations.relu(x, alpha=0.05)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128)(x)
        x = keras.activations.relu(x, alpha=0.05)
        return x

    def _get_final_classifier(features):
        x = layers.Dense(384)(features)
        x = keras.activations.relu(x, alpha=0.05)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(256)(x)
        x = keras.activations.relu(x, alpha=0.05)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(128)(x)
        x = keras.activations.relu(x, alpha=0.05)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(N_TARGETS, activation='sigmoid')(x)
        return x

    text_inputs = {col: keras.Input(shape=(), dtype=tf.string, name=f'{col}_text') for col in TEXT_COLS + UNORDERED_CATEGORIES}
    unordered_cat_inputs = {col: keras.Input(shape=(None,), dtype=tf.string, name=f'{col}_unordered_cat') for col in UNORDERED_CATEGORIES}
    ordered_cat_input = keras.Input(shape=(len(ORDERED_CATEGORIES)), name='ordered_cat_input')

    ordered_cat_features = [_get_ordered_category_model(ordered_cat_input)]
    text_features = [_get_text_model(text_inputs[col]) for col in TEXT_COLS + [POSITION]]
    unordered_cat_features = [_get_unordered_category_mode(unordered_cat_inputs[col], encoders[col]) for col in UNORDERED_CATEGORIES]

    features = layers.Concatenate()(ordered_cat_features + text_features + unordered_cat_features)
    out = _get_final_classifier(features)

    model = keras.Model(
        inputs=[ordered_cat_input] + list(text_inputs.values()) + list(unordered_cat_inputs.values()),
        outputs=out
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            tfa.metrics.F1Score(num_classes=9, average='micro', name='f1_score_micro'),
            tfa.metrics.F1Score(num_classes=9, average='macro', name='f1_score_macro')]
)

    return model
