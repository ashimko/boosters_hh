import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import torch
from torch import nn

from tqdm import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from config import PREPARED_DATA_PATH
from model_config import RANDOM_STATE, N_SPLITS, MODEL_NAME, BATCH_SIZE, LR, TOKENIZER_NAME
from model import CustomDataset, Classifier, train_model
from my_torch_utils import get_device
from helper import get_checkpoint_path, save_metric_plots, save_metrics, save_predictions, save_treshold
from evaluate import get_cv_results, get_one_opt_treshold

from models.pytorch_1.model_config import EPOCHS

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

        X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
        y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]

        tokenizer = AutoTokenizer.from_pretrained(f"cointegrated/{TOKENIZER_NAME}")

        train_set = CustomDataset(data=X_train, target=y_train, tokenizer=tokenizer)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle = True)
        
        val_set = CustomDataset(data=X_val, target=y_val, tokenizer=tokenizer)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle = False)

        device = get_device()
        model = Classifier()
        model.to(device)

        optimizer = torch.optim.Adam(params =  model.parameters(), lr=LR)

        checkpoint_path = get_checkpoint_path(MODEL_NAME, fold, ext='pt')
        best_model_path = get_checkpoint_path(MODEL_NAME, 'best_model', ext='pt')

        val_targets=[]
        val_outputs=[] 
        model, val_pred_proba = train_model(
            start_epochs=1, 
            n_epochs=EPOCHS, 
            val_loss_min_input=np.Inf, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            model=model, 
            optimizer=optimizer,
            checkpoint_path=checkpoint_path,
            best_model_path=best_model_path,
            val_targets=val_targets,
            val_outputs=val_outputs,
            device=device,
            tokenizer=tokenizer
        )

        oof_pred_proba.iloc[val_idx, :] = val_pred_proba

    print('getting best treshold...')
    opt_treshold = get_one_opt_treshold(target, oof_pred_proba)
    save_treshold(opt_treshold)
    oof_pred_labels.loc[:, :] = np.where(oof_pred_proba.values >= opt_treshold, 1, 0)

    save_predictions(oof_pred_proba, 'oof', MODEL_NAME, 'pred_proba')
    save_predictions(oof_pred_labels, 'oof', MODEL_NAME, 'pred_labels')
    
    cv_results = get_cv_results(target, oof_pred_labels, oof_pred_proba)
    save_metrics(cv_results, MODEL_NAME)
    save_metric_plots(target, oof_pred_proba, MODEL_NAME)


if __name__ == '__main__':
    fit()
