import os
from typing import DefaultDict, Dict, List
from collections import defaultdict

import numpy as np
import tensorflow as tf
from pandas import DataFrame, Series
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, log_loss, recall_score, precision_score
from config import MODEL_PATH, N_LABELS
from model import BATCH_SIZE, get_model_input, N_EPOCHS

from helper import save_model

def update_with_label_metrics(y_true: DataFrame, y_pred: DataFrame, 
                      cv_results: DefaultDict, mode: str = 'test') -> DataFrame:
    cv_results[f'{mode}_f1_samples'].append(f1_score(y_true, y_pred, average='samples', zero_division=0))
    cv_results[f'{mode}_recall_samples'].append(recall_score(y_true, y_pred, average='samples'))
    cv_results[f'{mode}_precision_samples'].append(precision_score(y_true, y_pred, average='samples'))
    return cv_results

def update_with_pred_proba_metrics(y_true: DataFrame, y_pred: DataFrame, 
                      cv_results: DefaultDict, mode: str = 'test') -> DataFrame:
    cv_results[f'{mode}_neg_log_loss'].append(-log_loss(y_true, y_pred))
    cv_results[f'{mode}_roc_auc_ovo'].append(roc_auc_score(y_true, y_pred, average='samples', multi_class='ovo'))
    return cv_results


def get_pred_labels(pred_proba: np.ndarray, y_true: np.ndarray = None) -> np.ndarray:
    pred_labels = np.zeros_like(pred_proba, dtype=np.int8)
    for col_idx in range(N_LABELS):
        thres, _ = optimal_threshold(pred_proba[:, col_idx])
        pred_labels[:, col_idx] = np.where(pred_proba[:, col_idx] > thres, 1, 0)
    if y_true is not None:
        print('f1_scores_samples', f1_score(y_true, pred_labels, average='samples', zero_division=0))
    return pred_labels


def expect_f1(y_prob, thres):
    idxs = np.where(y_prob >= thres)[0]
    tp = y_prob[idxs].sum()
    fp = len(idxs) - tp
    idxs = np.where(y_prob < thres)[0]
    fn = y_prob[idxs].sum()
    return 2*tp / (2*tp + fp + fn)


def optimal_threshold(y_prob):
    y_prob = np.sort(y_prob)[::-1]
    f1s = [expect_f1(y_prob, p) for p in y_prob]
    thres = y_prob[np.argmax(f1s)]
    return thres, f1s


def evaluate(model: Pipeline, train: DataFrame, target: Series, test: DataFrame, 
             n_splits: int, random_state: int = 42) -> Dict:
    cv = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_pred_labels = DataFrame(data=np.zeros_like(target, dtype=np.int8), index=target.index, columns=target.columns)
    oof_pred_proba = DataFrame(data=np.zeros_like(target, dtype=np.float32), index=target.index, columns=target.columns)
    
    test_shape = (len(test), target.shape[1])
    ens_test_pred_labels = DataFrame(data=np.zeros(shape=test_shape, dtype=np.int8), index=test.index, columns=target.columns)
    ens_test_pred_proba = DataFrame(data=np.zeros(shape=test_shape, dtype=np.float32), index=test.index, columns=target.columns)


    cv_results = defaultdict(list)
    fold = 0
    for train_idx, val_idx in cv.split(X=train, y=target):
        print(f'fold {fold}')
        X_train, y_train = train.iloc[train_idx], target.iloc[train_idx]
        X_val, y_val  = train.iloc[val_idx], target.iloc[val_idx]


        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_f1_score_micro',
            mode='max',
            patience=3)
            
        checkpoint_filepath = os.path.join(MODEL_PATH, f'fold_{fold}_checkpoint')
        checkopoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_best_only=True,
            monitor='val_f1_score_micro',
            mode='max',
            verbose=1
        )

        # model.fit(
        #     x=get_model_input(X_train), 
        #     y=y_train, 
        #     epochs=N_EPOCHS, 
        #     batch_size=BATCH_SIZE,
        #     validation_split=0.15, 
        #     callbacks=[early_stopping, checkopoint])
        
        model.load_weights(checkpoint_filepath).expect_partial()


        pred_proba_val = model.predict(get_model_input(X_val))
        pred_proba_train = model.predict(get_model_input(X_train))

        oof_pred_labels.iloc[val_idx] = get_pred_labels(pred_proba_val, y_val)
        cv_results = update_with_label_metrics(
            y_true=target.iloc[val_idx], 
            y_pred=oof_pred_labels.iloc[val_idx],
            cv_results=cv_results, mode='val')
        cv_results = update_with_label_metrics(
            y_true=target.iloc[train_idx], 
            y_pred=get_pred_labels(pred_proba_train, y_train),
            cv_results=cv_results, mode='train')

        oof_pred_proba.iloc[val_idx] = pred_proba_val
        cv_results = update_with_pred_proba_metrics(
            y_true=target.iloc[val_idx], 
            y_pred=oof_pred_labels.iloc[val_idx],
            cv_results=cv_results, mode='val')

        cv_results = update_with_pred_proba_metrics(
            y_true=target.iloc[train_idx], 
            y_pred=pred_proba_train,
            cv_results=cv_results, mode='train')
        
        pred_proba_test = model.predict(get_model_input(test))
        ens_test_pred_labels += get_pred_labels(pred_proba_test)
        ens_test_pred_proba += pred_proba_test
        fold += 1
    

    checkpoint_filepath = os.path.join(MODEL_PATH, 'whole_train_checkpoint')
    checkopoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_best_only=True,
        monitor='val_f1_score_micro',
        mode='max',
        verbose=1
    )
        
    # model.fit(
    #     x=get_model_input(train), 
    #     y=target, 
    #     epochs=N_EPOCHS, 
    #     batch_size=BATCH_SIZE,
    #     validation_split=0.15, 
    #     callbacks=[early_stopping, checkopoint])

    model.load_weights(checkpoint_filepath).expect_partial()
    pred_proba = model.predict(get_model_input(test))
    pred_labels = get_pred_labels(pred_proba)
    ens_test_pred_labels += pred_labels
    whole_train_test_pred_labels =  DataFrame(data=pred_labels, index=test.index, columns=target.columns)
    
    ens_test_pred_proba += pred_proba
    whole_train_test_pred_proba = DataFrame(data=pred_proba, index=test.index, columns=target.columns)
        
    ens_test_pred_labels /= (n_splits + 1)
    ens_test_pred_proba /= (n_splits + 1)

    return (cv_results, oof_pred_labels, oof_pred_proba, 
            ens_test_pred_labels, ens_test_pred_proba, 
            whole_train_test_pred_labels, whole_train_test_pred_proba)
