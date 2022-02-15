from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub
import tensorflow_text as text
from config import (MORPH_TAG_COLS, NORMALIZED_TEXT_COLS, ORDERED_CATEGORIES,
                    POSITION, TEXT_COLS, UNORDERED_CATEGORIES)
from tensorflow import keras
from tensorflow.keras import layers

from model_config import N_TARGETS, VOCAB_SIZE, TEXT_ENC_COLS


def get_encoders(data: pd.DataFrame, **keywords) -> Dict:
    encoders = {}
    for col in data.columns:
        encoder = layers.TextVectorization(max_tokens=VOCAB_SIZE, name=col, **keywords)
        encoder.adapt(data[col])
        encoders[col] = encoder
    return encoders


def get_model_input(data: pd.DataFrame) -> Dict:
    x = {f'{col}_text': data[col] for col in TEXT_ENC_COLS}
    x.update({f'{col}_unordered_cat': data[col] for col in UNORDERED_CATEGORIES})
    x.update({'ordered_cat_input': data[ORDERED_CATEGORIES]})
    return x


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
def soft_f1_samples_metric(y, y_hat):
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
    y_hat = tf.round(y_hat)
    tp = tf.reduce_sum(y_hat * y, axis=1)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=1)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=1)
    soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    samples_cost = tf.reduce_mean(soft_f1) # average on all labels
    return samples_cost


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


def get_model(encoders: Dict) -> keras.Model:

    def _get_text_model(text_input):
        preprocessor = hub.KerasLayer(
           "https://tfhub.dev/jeongukjae/smaller_LaBSE_15lang_preprocess/1")
        encoder_inputs = preprocessor(text_input)
        encoder = hub.KerasLayer(
            "https://tfhub.dev/jeongukjae/smaller_LaBSE_15lang/1",
            trainable=False)
        outputs = encoder(encoder_inputs)
        pooled_output = outputs["pooled_output"]      # [batch_size, 768].
        sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 768].
        rnn_out = layers.LSTM(32, return_sequences=True)(sequence_output)
        max_pool_transformer = layers.GlobalMaxPool1D()(sequence_output)
        max_pool_rnn = layers.GlobalMaxPool1D()(rnn_out)
        avg_pool_rnn = layers.GlobalAveragePooling1D()(rnn_out)
        text_features = layers.Concatenate()(
            [pooled_output, max_pool_transformer, max_pool_rnn, avg_pool_rnn])
        return text_features

    def _get_ordered_category_model(input):
        x = layers.Dense(128)(input)
        x = keras.activations.relu(x, alpha=0.1)
        return x

    def _get_unordered_category_mode(input, encoder):
        x = encoder(input)
        x = layers.CategoryEncoding(num_tokens=len(encoder.get_vocabulary()))(x)
        x = layers.Dense(128)(x)
        x = keras.activations.relu(x, alpha=0.1)
        return x

    def _get_final_classifier(features):
        x = layers.Dense(512)(features)
        x = keras.activations.relu(x, alpha=0.1)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(256)(x)
        x = keras.activations.relu(x, alpha=0.1)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(128)(x)
        x = keras.activations.relu(x, alpha=0.1)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(N_TARGETS, activation='sigmoid')(x)
        return x

    text_inputs = {col: keras.Input(shape=(), dtype=tf.string, name=f'{col}_text') for col in TEXT_ENC_COLS}
    unordered_cat_inputs = {col: keras.Input(shape=(None,), dtype=tf.string, name=f'{col}_unordered_cat') for col in UNORDERED_CATEGORIES}
    ordered_cat_input = keras.Input(shape=(len(ORDERED_CATEGORIES)), name='ordered_cat_input')

    ordered_cat_features = [_get_ordered_category_model(ordered_cat_input)]
    text_features = [_get_text_model(text_inputs[col]) for col in TEXT_COLS + NORMALIZED_TEXT_COLS]
    unordered_cat_features = [_get_unordered_category_mode(unordered_cat_inputs[col], encoders[col]) for col in UNORDERED_CATEGORIES]

    features = layers.Concatenate()(ordered_cat_features + text_features + unordered_cat_features)
    out = _get_final_classifier(features)

    model = keras.Model(
        inputs=[ordered_cat_input] + list(text_inputs.values()) + list(unordered_cat_inputs.values()),
        outputs=out
    )

    model.compile(
        optimizer=Ranger(), # keras.optimizers.Adam(learning_rate=0.0003),
        loss=keras.losses.BinaryCrossentropy(from_logits=False), #  soft_f1_samples_loss,
        metrics=soft_f1_samples_metric
)

    return model
