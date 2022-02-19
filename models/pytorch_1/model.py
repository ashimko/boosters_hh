import transformers
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
from model_config import TOKENIZER_NAME, MAX_SEQ_LEN, MODEL_NAME
from config import TEXT_COLS
from my_torch_utils import save_ckp
from sklearn.metrics import f1_score


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


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

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=MAX_SEQ_LEN,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets.iloc[index], dtype=torch.float)
        }


class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.encoder = transformers.BertModel.from_pretrained(MODEL_NAME)
        self.dropout = torch.nn.Dropout(0.3)
        self.out = torch.nn.Linear(768, 9)
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.encoder(ids, attention_mask = mask, token_type_ids = token_type_ids)
        output_2 = self.dropout(output_1)
        output = self.out(output_2)
        return output


def train_model(start_epochs,  n_epochs, val_loss_min_input, 
          train_loader, val_loader, model, 
          optimizer, checkpoint_path, best_model_path,
          device, val_targets, val_outputs):
   
  # initialize tracker for minimum validation loss
  valid_loss_min = val_loss_min_input 
   
 
  for epoch in range(start_epochs, n_epochs+1):
    train_loss = 0
    valid_loss = 0

    model.train()
    print('############# Epoch {}: Training Start   #############'.format(epoch))
    for batch_idx, data in enumerate(train_loader):
        #print('yyy epoch', batch_idx)
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        #if batch_idx%5000==0:
         #   print(f'Epoch: {epoch}, Training Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print('before loss data in training', loss.item(), train_loss)
        train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
        #print('after loss data in training', loss.item(), train_loss)
    
    print('############# Epoch {}: Training End     #############'.format(epoch))
    
    print('############# Epoch {}: Validation Start   #############'.format(epoch))
    ######################    
    # validate the model #
    ######################
 
    model.eval()
   
    with torch.no_grad():
      for batch_idx, data in enumerate(val_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)

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
        
      ## TODO: save the model if validation loss has decreased
      if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
        # save checkpoint as best model
        save_ckp(checkpoint, True, checkpoint_path, best_model_path)
        valid_loss_min = valid_loss

    print('############# Epoch {}  Done   #############\n'.format(epoch))

  return model, val_outputs