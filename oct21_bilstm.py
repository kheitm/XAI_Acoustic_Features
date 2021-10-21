
# %%
# imported packages
import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import json
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy
from statistics import mean


# %%
# Prepared DATASET
class JSON_Dataset():
    def __init__(self, FilePath): 
        self.FilePath = FilePath 
        
        with open(FilePath) as fileinput: 
            XY = json.load(fileinput) 
        X = np.array([Row[1] for Row in XY])
        Y = np.array([Row[0] for Row in XY])
        self.data = (Y, X)
        self.len = len(self.data[0]) 
        self.shape = self.data[0].shape 
        
    def __getitem__(self, index):
        Xi = self.data[1][index] 
        Yi = self.data[0][index] 
        return Xi, Yi 

    def __len__(self):
        return self.len #Setting length attribute
    
    def __shape__(self): #Setting shape attribute
        return self.shape

# %%
# called model Class and associated functions
def reset_weights(model):
    if isinstance(model, nn.LSTM) or isinstance(model, nn.Linear):
        model.reset_parameters()

def binary_accuracy(prediction, target):
    preds = torch.round(prediction) # round to the nearest integer
    correct = (preds == target).float()
    accuracy = correct.sum()/len(correct)
    return accuracy


class biLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, label_size, dropout):
        super(biLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.3)
        self.attn = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=1)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, label_size)

        
    def forward(self, x):
        out,(hidden,_) = self.lstm(x)
        out = self.attn(hidden)
        out = out.mean(dim=0)
        out = self.fc1(out)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.fc2(out)
        # out = self.relu(out)
        return out
# %%
def train_epoch(model,device, trainloader, loss_fn, optimizer):
    train_loss, train_acc = 0, 0
    model.train()
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.float)
        targets = targets.unsqueeze(1)
        optimizer.zero_grad()
        prediction = model(inputs)
        loss = loss_fn(prediction, targets)
        accuracy = binary_accuracy(prediction, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += accuracy.item()
    return train_loss/len(trainloader), train_acc/len(trainloader)*100

def valid_epoch(model, device, validloader, loss_fn):
    valid_loss, valid_acc = 0, 0
    model.eval()
    with torch.no_grad():
      for inputs, targets in validloader:
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.float)
            targets = targets.unsqueeze(1)
            prediction = model(inputs)
            loss = loss_fn(prediction, targets)
            accuracy =  binary_accuracy(prediction, targets)
            valid_loss += loss.item()
            valid_acc += accuracy.item()
      return valid_loss/len(validloader), valid_acc/len(validloader)*100

# %%
# FIT MODEL

def fit(dataset,input_size, hidden_size, num_layers, label_size, dropout, 
        learning_rate, num_epochs,device, loss_fn, skf):
    max_valid_accuracy = []
    mean_valid_accuracy =[]
    for fold, (train_ids, valid_ids) in enumerate(skf.split(dataset.data[1], dataset.data[0])):
        print(f'\n FOLD {fold + 1}')
        print('---------------------------------------------------------------------------')
        train_subsampler = SubsetRandomSampler(train_ids)
        valid_subsampler = SubsetRandomSampler(valid_ids)
        trainloader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        validloader = DataLoader(dataset, batch_size=batch_size, sampler=valid_subsampler)
    
        model = biLSTM(input_size, hidden_size, num_layers, label_size, dropout).to(device)
        model = model.float()
        model.apply(reset_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        valid_loss_min = np.Inf 
        early_stopping_iter = 300
        early_stopping_counter = 0

        valid_accuracy = []
        for epoch in range(num_epochs):
            print(f'Starting epoch {epoch+1}')

            train_loss, train_acc = train_epoch(model, device, trainloader,loss_fn, optimizer)
            valid_loss, valid_acc = valid_epoch(model, device, validloader, loss_fn) 

            if  valid_loss < valid_loss_min:
                valid_loss_min = valid_loss
                best_model_state = deepcopy(model.state_dict())
                torch.save(best_model_state, f'./saved_models/model-fold-{fold +1}.pth')
                # torch.save(model.state_dict(), f'./saved_models/model-fold-{fold +1}.pth')
            else:
                early_stopping_counter += 1

            if early_stopping_counter > early_stopping_iter:  # if not improving for n iterations then stop
                break

            print("Epoch:{}/{} Training Loss:{:.3f}  Validation Loss:{:.3f} Training Accuracy {:.2f}% Validation Accuracy {:.2f}%".format(epoch + 1, num_epochs, train_loss, valid_loss, \
                train_acc, valid_acc))
            
            valid_accuracy.append(valid_acc)

        max_valid_accuracy.append(max(valid_accuracy))
        mean_valid_accuracy.append(mean(valid_accuracy)) 

  
    # return valid_loss_min, print(max_valid_accuracy)
    return valid_loss_min, print("Max Accuracy: Fold_1:{}, Fold_2:{}, Fold_3:{}, Fold_4:{}, Fold_5:{}".format(max_valid_accuracy[0], 
               max_valid_accuracy[1], max_valid_accuracy[2], max_valid_accuracy[3], max_valid_accuracy[4])), print("Mean Accuracy: Fold_1:{}, \
               Fold_2:{}, Fold_3:{}, Fold_4:{}, Fold_5:{}".format(max_valid_accuracy[0], mean_valid_accuracy[1], mean_valid_accuracy[2], \
                   mean_valid_accuracy[3], mean_valid_accuracy[4])) 

# %%
# SET MODEL HYPER-PARAMETERS
torch.manual_seed(42)
k_folds = 5
num_epochs = 300
input_size = 25 # #number of features in input
num_layers = 2
hidden_size = 25 #number of features in hidden state
label_size = 1
learning_rate = 0.001
batch_size = 128
dropout = 0.5
loss_fn = nn.BCEWithLogitsLoss()
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
filepath = '/mount/arbeitsdaten/thesis-dp-1/heitmekn/working/ad.json'
dataset = JSON_Dataset(filepath)
fit(dataset,input_size, hidden_size, num_layers, label_size, dropout, 
        learning_rate, num_epochs,device, loss_fn, skf)

