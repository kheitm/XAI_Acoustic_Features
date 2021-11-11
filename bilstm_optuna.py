
# %%
# optimising network

import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from datetime import datetime
import optuna
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_epochs = 300
train_dataset = '/mount/arbeitsdaten/thesis-dp-1/heitmekn/working/train_ad.json'


class AD_Dataset():
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
   
# called model Class and associated functions
def load_data(filepath, batch_size):
    dataset = AD_Dataset(filepath)
    train_size = int(len(dataset) * 0.7)
    valid_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, valid_size])
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
    validloader = torch.utils.data.DataLoader(val_data, batch_size = batch_size, shuffle = True) #Passing the trainset into a Dataloader
    return trainloader, validloader

def reset_weights(model):
    if isinstance(model, nn.LSTM) or isinstance(model, nn.Linear):
        model.reset_parameters()

def binary_accuracy(prediction, target):
    preds = torch.round(prediction) # round to the nearest integer
    correct = (preds == target).float()
    accuracy = correct.sum()/len(correct)
    return accuracy

  

class biLSTM(nn.Module):
    def __init__(self, trial):
        super(biLSTM, self).__init__()
        self.lstm = nn.LSTM(25, 25, num_layers=2, batch_first=True, bidirectional=True)
        self.attn = nn.TransformerEncoderLayer(d_model=25, nhead=1)
        #self.attn = nn.MultiheadAttention(25, 1)
        self.relu = nn.ReLU()
        dropout_rate = trial.suggest_float("dropout_rate", 0, 0.7,step=0.1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(25, 25)
        self.fc2 = nn.Linear(25, 1)

        
    def forward(self, x):
        out,(hidden,_) = self.lstm(x)
        out = self.attn(hidden)
        out = out.mean(dim=0)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class biLSTM(nn.Module):
    def __init__(self, trial):
        super(biLSTM, self).__init__()
        self.lstm = nn.LSTM(25, 25, num_layers=2, batch_first=True, bidirectional=True)
        self.attn = nn.TransformerEncoderLayer(d_model=25, nhead=1)
        self.relu = nn.ReLU()
        dropout_rate = trial.suggest_float("dropout_rate", 0, 0.7,step=0.1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(25, 25)
        self.fc2 = nn.Linear(25, 1)

        
    def forward(self, x):
        out,(hidden,_) = self.lstm(x)
        out = self.attn(hidden)
        out = out[-1, :, :]
        out = self.fc1(out)
        out = self.relu(out)
        # out = self.fc2(out)[-1]
        out = self.fc2(out)
        return out

def objective(trial):

    # Generate the model.

    model = biLSTM(trial).to(device)

    # optimizer_name = trial.suggest_categorical("optimizer", ["RMSprop", "SGD"])
    # momentum = trial.suggest_float("momentum", 0.0, 1.0)
    # lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    # optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr,momentum=momentum)
    # batch_size = trial.suggest_int("batch_size",16,256, step=16)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "Adadelta","Adagrad"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1,log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    batch_size =trial.suggest_int("batch_size", 16, 256,step=16)


    loss_fn = nn.BCEWithLogitsLoss()
    trainloader, validloader = load_data(train_dataset, batch_size=batch_size)

    # training of the model
    for epoch in range(num_epochs):
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

        model.eval()
        valid_acc = 0
        valid_loss = 0.0
        with torch.no_grad():
            for inputs, targets in validloader:
                    inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.float)
                    targets = targets.unsqueeze(1)
                    prediction = model(inputs)
                    loss = loss_fn(prediction, targets)
                    acc =  binary_accuracy(prediction, targets) 
                    valid_loss += loss.item()
                    valid_acc += acc.item()
        
        # final_loss = valid_loss/len(validloader.dataset)
        accuracy = valid_acc/len(validloader)
        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

trial = study.best_trial

print('Accuracy: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))

df = study.trials_dataframe().drop(['state','datetime_start','datetime_complete','duration','number'], axis=1)
path = "/mount/arbeitsdaten/thesis-dp-1/heitmekn/working/saved_trials/bilstm_{}"
path = path.format(datetime.now().replace(microsecond=0).isoformat())
df.to_csv(path + ".csv")


