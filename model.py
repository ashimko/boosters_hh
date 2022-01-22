import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_text as text
import keras.backend as K
from typing import Dict
import tensorflow_hub as hub
from config import POSITION, TEXT_COLS, ORDERED_CATEGORIES, UNORDERED_CATEGORIES


VOCAB_SIZE = 15000
BATCH_SIZE = 1024
N_TARGETS = 9
N_EPOCHS = 100


def get_model_input(data: pd.DataFrame) -> Dict:
    x = {f'{col}_text': data[col] for col in TEXT_COLS + UNORDERED_CATEGORIES}
    x.update({f'{col}_unordered_cat': data[col] for col in UNORDERED_CATEGORIES})
    x.update({'ordered_cat_input': data[ORDERED_CATEGORIES]})
    return x


import tensorflow_addons as tfa


def Ranger(sync_period=6,
           slow_step_size=0.5,
           learning_rate=0.001,
           beta_1=0.9,
           beta_2=0.999,
           epsilon=1e-7,
           weight_decay=0.,
           amsgrad=False,
           sma_threshold=5.0,
           total_steps=0,
           warmup_proportion=0.1,
           min_lr=0.,
           name="Ranger"):
    inner = tfa.optimizers.RectifiedAdam(learning_rate, beta_1, beta_2, epsilon, weight_decay, amsgrad, sma_threshold, total_steps, warmup_proportion, min_lr, name)
    optim = tfa.optimizers.Lookahead(inner, sync_period, slow_step_size, name)
    return optim


@tf.function
def soft_f1_samples_loss(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=1)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=1)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=1)
    soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    cost = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
    samples_cost = tf.reduce_mean(cost) # average on all labels
    return samples_cost


def make_model(encoders: Dict) -> keras.Model:

    def _get_text_model(text_input):
        # preprocessor = hub.KerasLayer(
        #      "https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2")
        # encoder_inputs = preprocessor(text_input)
        encoder = hub.KerasLayer(
            "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3",
            trainable=False)
        outputs = encoder(text_input)
        # pooled_output = outputs["pooled_output"]
        # sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 768].
        x = layers.Dense(128)(outputs)
        x = tfa.layers.GELU()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128)(x)
        return x

    def _get_ordered_category_model(input):
        x = layers.Dense(128)(input)
        x = tfa.layers.GELU()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128)(x)
        return x

    def _get_unordered_category_mode(input, encoder):
        x = encoder(input)
        x = layers.CategoryEncoding(num_tokens=len(encoder.get_vocabulary()))(x)
        x = layers.Dense(128)(x)
        x = tfa.layers.GELU()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128)(x)
        x = tfa.layers.GELU()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128)(x)
        return x

    def _get_final_classifier(features):
        x = layers.Dense(256)(features)
        x = tfa.layers.GELU()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128)(x)
        x = tfa.layers.GELU()(x)
        x = layers.Dropout(0.3)(x)
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
        optimizer=Ranger(),
        loss=soft_f1_samples_loss,
        metrics=[
            tfa.metrics.F1Score(num_classes=9, average='micro', name='f1_score_micro'),
            tfa.metrics.F1Score(num_classes=9, average='macro', name='f1_score_macro')]
)

    return model
