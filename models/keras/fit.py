import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import f1_score
from config import *
from utils import squeeze_pred_proba
from evaluate import get_pred_labels, get_cv_results
from helper import save_metric_plots, save_metrics, save_predictions, get_checkpoint_path

from model import get_model, get_model_input, get_encoders
from model_config import MODEL_NAME, N_EPOCHS, BATCH_SIZE


def fit():
    train = pd.read_pickle(os.path.join(PREPARED_DATA_PATH, 'train.pkl'))

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
        
        encoders = get_encoders(train[TEXT_COLS+UNORDERED_CATEGORIES])
        model = get_model(encoders)

        X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
        y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_soft_f1_samples_metric',
            mode='max',
            patience=2)
            
        checkpoint_filepath = get_checkpoint_path(MODEL_NAME, fold)
        checkopoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
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

        print(X_train.shape)
        model.fit(
            x=get_model_input(X_train), 
            y=y_train, 
            epochs=N_EPOCHS, 
            batch_size=BATCH_SIZE,
            validation_split=0.15,
            validation_batch_size=BATCH_SIZE,
            callbacks=[early_stopping, checkopoint, lr_scheduler])
        
        model = load_model(checkpoint_filepath)

        val_pred_proba = squeeze_pred_proba(model.predict_proba(get_model_input(X_val)))
        oof_pred_proba.iloc[val_idx] = val_pred_proba

        val_pred_labels = get_pred_labels(val_pred_proba)
        oof_pred_labels.iloc[val_idx] = val_pred_labels
        
        score = f1_score(y_val, val_pred_labels, average='samples', zero_division=0)
        print(f'model name {MODEL_NAME}, fold {fold}, f1_score: {score}')
        

    save_predictions(oof_pred_proba, 'oof', MODEL_NAME, 'pred_proba')
    save_predictions(oof_pred_labels, 'oof', MODEL_NAME, 'pred_labels')
    
    cv_results = get_cv_results(target, oof_pred_labels, oof_pred_proba)
    save_metrics(cv_results, MODEL_NAME)
    save_metric_plots(target, oof_pred_proba, MODEL_NAME)


if __name__ == '__main__':
    fit()
