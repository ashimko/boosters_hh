import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import tensorflow as tf
from config import PREPARED_DATA_PATH
from evaluate import get_cv_results, get_one_opt_treshold, get_pred_labels
from helper import (get_checkpoint_path, save_metric_plots, save_metrics,
                    save_predictions, save_treshold)
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import f1_score

from model import get_encoders, get_model, get_model_input
from model_config import (BATCH_SIZE, MODEL_NAME, N_EPOCHS, N_SPLITS,
                          RANDOM_STATE, TEXT_ENC_COLS)
from data_tools import get_train


def fit():
    train = get_train()
    target = pd.read_pickle(os.path.join(PREPARED_DATA_PATH, 'target.pkl'))
    cv = MultilabelStratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof_pred_proba = pd.DataFrame(
        data=np.zeros(shape=target.shape),
        columns=target.columns,
        index=target.index
    )
    oof_pred_labels = oof_pred_proba.copy()
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X=train, y=target)):
        print(f'start training {MODEL_NAME}, fold {fold}...')
        
        encoders = get_encoders(train[TEXT_ENC_COLS])
        model = get_model(encoders)

        X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
        y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_soft_f1_samples_metric',
            mode='max',
            patience=3)
            
        checkpoint_filepath = get_checkpoint_path(MODEL_NAME, fold)
        checkopoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            save_best_only=True,
            monitor='val_soft_f1_samples_metric',
            mode='max',
            verbose=1
        )

        def scheduler(epoch, lr):
            if epoch < 1:
                return lr
            else:
                return lr * tf.math.exp(-0.1)
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

        model.fit(
            x=get_model_input(X_train), 
            y=y_train, 
            epochs=N_EPOCHS, 
            batch_size=BATCH_SIZE,
            validation_split=0.15,
            validation_batch_size=BATCH_SIZE,
            callbacks=[early_stopping, checkopoint, lr_scheduler])
        
        model.load_weights(checkpoint_filepath).expect_partial()

        val_pred_proba = model.predict(get_model_input(X_val))
        oof_pred_proba.iloc[val_idx, :] = val_pred_proba

        val_pred_labels = get_pred_labels(val_pred_proba)
        oof_pred_labels.iloc[val_idx, :] = val_pred_labels
        
    print('getting best treshold...')
    opt_treshold = get_one_opt_treshold(target, oof_pred_proba)
    save_treshold(opt_treshold)
    oof_pred_labels.loc[:, :] = np.where(oof_pred_proba.values >= opt_treshold, 1, 0)    

    save_predictions(oof_pred_proba, 'oof', MODEL_NAME, 'pred_proba')
    save_predictions(oof_pred_labels, 'oof', MODEL_NAME, 'pred_labels')
    
    cv_results = get_cv_results(target, oof_pred_labels, oof_pred_proba)
    save_metrics(cv_results, MODEL_NAME)
    save_metric_plots(target, oof_pred_proba, MODEL_NAME)

    print('start fiting on whole train...')

    early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_soft_f1_samples_metric',
            mode='max',
            patience=3)
            
    checkpoint_filepath = get_checkpoint_path(MODEL_NAME, -1)
    checkopoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_soft_f1_samples_metric',
        mode='max',
        verbose=1
    )

    def scheduler(epoch, lr):
        if epoch < 1:
            return lr
        else:
            return lr * tf.math.exp(-0.1)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

    model.fit(
        x=get_model_input(train), 
        y=target, 
        epochs=N_EPOCHS, 
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        validation_batch_size=BATCH_SIZE,
        callbacks=[early_stopping, checkopoint, lr_scheduler])


if __name__ == '__main__':
    fit()
