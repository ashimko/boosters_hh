import copy
import gc
import random
import re
import time

import numpy as np
import pandas as pd
import torch
from torchtext import data
#import spacy
from tqdm import tnrange, tqdm_notebook
from tqdm.auto import tqdm

tqdm.pandas(desc='Progress')
import os
from collections import Counter
from functools import partial
from multiprocessing import Pool

import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk import word_tokenize
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
# cross validation and metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchtext.data import Example
from unidecode import unidecode

from model_config import (BATCH_SIZE, DEBUG, EMBED_SIZE, MAX_FEATURES, MAXLEN,
                          N_EPOCHS, N_SPLITS, SEED)

loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')


def seed_everything(seed=10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()


# code inspired from: https://github.com/anandsaha/pytorch.cyclic.learning.rate/blob/master/cls.py
class CyclicLR(object):
    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs


class MyDataset(Dataset):
    def __init__(self,dataset):
        self.dataset = dataset
    def __getitem__(self,index):
        data,target = self.dataset[index]
        return data,target,index
    def __len__(self):
        return len(self.dataset)


def pytorch_model_run_cv(x_train,y_train,features,x_test, model_obj, feats = False,clip = True):
    seed_everything()
    avg_losses_f = []
    avg_val_losses_f = []
    # matrix for the out-of-fold predictions
    train_preds = np.zeros((len(x_train)))
    # matrix for the predictions on the test set
    test_preds = np.zeros((len(x_test)))
    splits = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(x_train, y_train))
    for i, (train_idx, valid_idx) in enumerate(splits):
        seed_everything(i*1000+i)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        if feats:
            features = np.array(features)
        x_train_fold = torch.tensor(x_train[train_idx.astype(int)], dtype=torch.long).cuda()
        y_train_fold = torch.tensor(y_train[train_idx.astype(int), np.newaxis], dtype=torch.float32).cuda()
        if feats:
            kfold_X_features = features[train_idx.astype(int)]
            kfold_X_valid_features = features[valid_idx.astype(int)]
        x_val_fold = torch.tensor(x_train[valid_idx.astype(int)], dtype=torch.long).cuda()
        y_val_fold = torch.tensor(y_train[valid_idx.astype(int), np.newaxis], dtype=torch.float32).cuda()
        
        model = copy.deepcopy(model_obj)

        model.cuda()

        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')

        step_size = 300
        base_lr, max_lr = 0.001, 0.003   
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                 lr=max_lr)
        
        ################################################################################################
        scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,
                   step_size=step_size, mode='exp_range',
                   gamma=0.99994)
        ###############################################################################################

        train = MyDataset(torch.utils.data.TensorDataset(x_train_fold, y_train_fold))
        valid = MyDataset(torch.utils.data.TensorDataset(x_val_fold, y_val_fold))
        
        train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=BATCH_SIZE, shuffle=False)

        print(f'Fold {i + 1}')
        for epoch in range(N_EPOCHS):
            start_time = time.time()
            model.train()

            avg_loss = 0.  
            for i, (x_batch, y_batch, index) in enumerate(train_loader):
                if feats:       
                    f = kfold_X_features[index]
                    y_pred = model([x_batch,f])
                else:
                    y_pred = model(x_batch)

                if scheduler:
                    scheduler.batch_step()

                # Compute and print loss.
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                if clip:
                    nn.utils.clip_grad_norm_(model.parameters(),1)
                optimizer.step()
                avg_loss += loss.item() / len(train_loader)
                
            model.eval()
            
            valid_preds_fold = np.zeros((x_val_fold.size(0)))
            test_preds_fold = np.zeros((len(x_test)))
            
            avg_val_loss = 0.
            for i, (x_batch, y_batch,index) in enumerate(valid_loader):
                if feats:
                    f = kfold_X_valid_features[index]            
                    y_pred = model([x_batch,f]).detach()
                else:
                    y_pred = model(x_batch).detach()
                
                avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
                valid_preds_fold[index] = sigmoid(y_pred.cpu().numpy())[:, 0]
            
            elapsed_time = time.time() - start_time 
            print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
                epoch + 1, N_EPOCHS, avg_loss, avg_val_loss, elapsed_time))
        avg_losses_f.append(avg_loss)
        avg_val_losses_f.append(avg_val_loss) 
        # predict all samples in the test set batch per batch
        for i, (x_batch,) in enumerate(test_loader):
            if feats:
                f = test_features[i * BATCH_SIZE:(i+1) * BATCH_SIZE]
                y_pred = model([x_batch,f]).detach()
            else:
                y_pred = model(x_batch).detach()

            test_preds_fold[i * BATCH_SIZE:(i+1) * BATCH_SIZE] = sigmoid(y_pred.cpu().numpy())[:, 0]
            
        train_preds[valid_idx] = valid_preds_fold
        test_preds += test_preds_fold / len(splits)

    print('All \t loss={:.4f} \t val_loss={:.4f} \t '.format(np.average(avg_losses_f),np.average(avg_val_losses_f)))
    return train_preds, test_preds