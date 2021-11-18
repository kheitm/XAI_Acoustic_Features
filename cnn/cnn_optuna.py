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


def binary_accuracy(prediction, target):
    preds = torch.round(prediction) # round to the nearest integer
    correct = (preds == target).float()
    accuracy = correct.sum()/len(correct)
    return accuracy

# %%
class ConvNet(nn.Module):
    def __init__(self, trial):
        super(ConvNet, self).__init__()
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 4)
        num_fc_layers = trial.suggest_int("num_fc_layers", 1, 2)

        self.layers = []
        input_depth = 993
        for i in range(num_conv_layers):
            output_depth = trial.suggest_int(f"conv_depth_{i}", 16, 64)
            self.layers.append(nn.Conv1d(input_depth, output_depth, kernel_size=5,  stride=(1),padding=(2)))
            self.layers.append(nn.ReLU())
            input_depth = output_depth
        self.layers.append(nn.MaxPool1d(2))
        p = trial.suggest_float(f"conv_dropout_{i}", 0.1, 0.4)
        self.layers.append(nn.Dropout(p))
        self.layers.append(nn.Flatten())

        input_feat = self._get_flatten_shape()
        for i in range(num_fc_layers):
            output_feat = trial.suggest_int(f"fc_output_feat_{i}", 16, 64)
            self.layers.append(nn.Linear(input_feat, output_feat))
            self.layers.append(nn.ReLU())
            p = trial.suggest_float(f"fc_dropout_{i}", 0.1, 0.4)
            self.layers.append(nn.Dropout(p))
            input_feat = output_feat
        self.layers.append(nn.Linear(input_feat, 1))
        self.layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*self.layers)        

    def _get_flatten_shape(self):
        conv_model = nn.Sequential(*self.layers)
        op_feat = conv_model(torch.rand(1, 993, 25))
        n_size = op_feat.data.view(1, -1).size(1)
        return n_size
 
    def forward(self, x):
        return self.model(x)


# %%
def objective(trial):

    # Generate the model.
    model = ConvNet(trial).to(device)

    # hyperparameter Option 1
    # optimizer_name = trial.suggest_categorical("optimizer", ["RMSprop", "SGD"])
    # momentum = trial.suggest_float("momentum", 0.0, 1.0)
    # lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    # optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr,momentum=momentum)
    # batch_size = trial.suggest_int("batch_size",16,128, step=16)

    # hyperparameter Option 2
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "Adadelta","Adagrad"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1,log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    batch_size=trial.suggest_int("batch_size", 16, 128,step=16)

    loss_fn = nn.BCELoss()
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
        
        accuracy = valid_acc/len(validloader)
        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


study = optuna.create_study(study_name="optimised CNN layers", direction='maximize')
study.optimize(objective, n_trials=700)

trial = study.best_trial

print('Accuracy: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))

df = study.trials_dataframe().drop(['state','datetime_start','datetime_complete','duration','number'], axis=1)
path = "/mount/arbeitsdaten/thesis-dp-1/heitmekn/working/saved_trials/cnn_layers_{}"
path = path.format(datetime.now().replace(microsecond=0).isoformat())
df.to_csv(path + ".csv")
