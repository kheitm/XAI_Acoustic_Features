
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import json
import pandas as pd

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

filepath = '/content/drive/MyDrive/thesis_data/subset20_lld_feats.json'
# filepath = '/content/drive/MyDrive/thesis_data/AD_lld_feats.json'
dataset = JSON_Dataset(filepath)

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
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # self.self_attention = nn.MultiheadAttention(input_size*2, num_heads=1, batch_first=True)
        self.attn = torch.nn.MultiheadAttention(embed_dim=25*2 , num_heads=1, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, label_size)
        self.sigmoid = nn.Sigmoid() 

        
    def forward(self, x):
        out,_ = self.lstm(x) 
        out = out[:, -1, :]
        # out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.sigmoid(out)
        return out

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
k_folds = 5
num_epochs = 1
input_size = 25 # #number of features in input
num_layers = 2
hidden_size = 25 #number of features in hidden state
label_size = 1
learning_rate = 0.001
batch_size = 10
dropout = 0.5
loss_fn = nn.BCELoss()
loss_fn = nn.BCEWithLogitsLoss()
results = {} # For fold results
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
kfold_dict = {}

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
    return train_loss, train_acc

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
      return valid_loss, valid_acc

for fold, (train_ids, valid_ids) in enumerate(skf.split(dataset.data[1], dataset.data[0])):
    print(f'\n FOLD {fold}')
    print('---------------------------------------------------------------------------')
    train_subsampler = SubsetRandomSampler(train_ids)
    valid_subsampler = SubsetRandomSampler(valid_ids)
    trainloader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
    validloader = DataLoader(dataset, batch_size=batch_size, sampler=valid_subsampler)
   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = biLSTM(input_size, hidden_size, num_layers, label_size, dropout).to(device)
    model = model.float()
    model.apply(reset_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # history = {'train_loss': [], 'valid_loss': [],'train_acc':[],'valid_acc':[]}

    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch+1}')

        train_loss, train_acc = train_epoch(model, device, trainloader,loss_fn, optimizer)
        valid_loss, valid_acc = valid_epoch(model, device, validloader, loss_fn) 

        epoch_train_loss = train_loss / len(trainloader.sampler)
        epoch_train_acc = train_acc / len(trainloader.sampler) * 100
        epoch_valid_loss = valid_loss / len(validloader.sampler)
        epoch_valid_acc = valid_acc / len(validloader.sampler) * 100

        print("Epoch:{}/{} Training Loss:{:.3f}  Validation Loss:{:.3f}  Validation Accuracy {:.2f}%".format(epoch + 1, num_epochs, epoch_train_loss, epoch_valid_loss, \
            epoch_train_acc, epoch_valid_acc))
        
        # history['train_loss'].append(train_loss)
        # history['valid_loss'].append(valid_loss)
        # history['train_acc'].append(train_acc)
        # history['valid_acc'].append(valid_acc)
        # kfold_dict['fold{}'.format(fold+1)] = history 

torch.save(model,'skf_oct16_1.pt')

