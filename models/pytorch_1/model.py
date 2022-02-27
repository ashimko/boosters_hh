import transformers
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
from model_config import TOKENIZER_NAME, MAX_SEQ_LEN, MODEL_NAME, EMB_DIM
from config import TEXT_COLS
from my_torch_utils import save_ckp
from sklearn.metrics import f1_score
from transformers import AutoModel


class CustomDataset(Dataset):

    def __init__(self, data, target, tokenizer):
        self.tokenizer = tokenizer
        self.data = data
        self.targets = target
        self.text_cols_idx = [i for i, c in enumerate(data.columns) if c in TEXT_COLS]


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        text = ' '.join(self.data.iloc[index, self.text_cols_idx])
        labels = torch.tensor(self.targets.iloc[index], dtype=torch.float)
        return text, labels


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def max_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    max_embeddings, _ = torch.max(token_embeddings * input_mask_expanded, 1)
    return max_embeddings


class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(f"cointegrated/{MODEL_NAME}")
        self.dropout = torch.nn.Dropout(0.3)
        self.out = torch.nn.Linear(EMB_DIM, 9)
    
    def forward(self, encoded_input):
        model_output = self.encoder(**encoded_input)
        mean_pool = mean_pooling(model_output, encoded_input['attention_mask'])
        output = self.out(mean_pool)
        return output


def train_model(start_epochs,  n_epochs, val_loss_min_input, 
          train_loader, val_loader, model, 
          optimizer, checkpoint_path, best_model_path,
          device, val_targets, val_outputs, tokenizer):
   
  # initialize tracker for minimum validation loss
  valid_loss_min = val_loss_min_input 
   
 
  for epoch in range(start_epochs, n_epochs+1):
    train_loss = 0
    valid_loss = 0

    model.train()
    print('############# Epoch {}: Training Start   #############'.format(epoch))
    for batch_idx, (sentences, targets) in enumerate(train_loader):
        #print('yyy epoch', batch_idx)
        encoded_input = tokenizer(list(sentences), padding=True, truncation=True, max_length=64, return_tensors='pt')
        encoded_input.to(device)
        outputs = model(encoded_input)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if batch_idx%1000==0:
           print(f'Epoch: {epoch}, Training Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('before loss data in training', loss.item(), train_loss)
        train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
        print('after loss data in training', loss.item(), train_loss)
    
    print('############# Epoch {}: Training End     #############'.format(epoch))
    
    print('############# Epoch {}: Validation Start   #############'.format(epoch))
    ######################    
    # validate the model #
    ######################
 
    model.eval()
   
    with torch.no_grad():
      for batch_idx, (sentences, targets) in enumerate(val_loader, 0):
            encoded_input = tokenizer(list(sentences), padding=True, truncation=True, max_length=64, return_tensors='pt')
            encoded_input.to(device)
            outputs = model(encoded_input)

            loss = loss_fn(outputs, targets)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
            val_target = targets.cpu().detach().numpy()
            val_targets.extend(val_target.tolist())
            val_output = torch.sigmoid(outputs).cpu().detach().numpy()
            val_outputs.extend(val_output.tolist())
            f1_samples_score = f1_score(val_target, np.where(val_output >= 0.5, 1, 0).astype(np.int8))

      print('############# Epoch {}: Validation End     #############'.format(epoch))
      # calculate average losses
      #print('before cal avg train loss', train_loss)
      train_loss = train_loss / len(train_loader)
      valid_loss = valid_loss / len(val_loader)
      # print training/validation statistics 
      print('Epoch: {} \tAvgerage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f} \tF1 samples: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss,
            f1_samples_score
            ))
      
      # create checkpoint variable and add important data
      checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
      }
        
      # save checkpoint
      save_ckp(checkpoint, False, checkpoint_path, best_model_path)
        
      
      if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
        # save checkpoint as best model
        save_ckp(checkpoint, True, checkpoint_path, best_model_path)
        valid_loss_min = valid_loss

    print('############# Epoch {}  Done   #############\n'.format(epoch))

  return model, val_outputs