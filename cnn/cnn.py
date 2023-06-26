
# %%
# imported packages
import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy



# %%
# loaded dataset
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
   
# %%
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

  
class ShallowCNN(nn.Module):
    def __init__(self):
        super(ShallowCNN, self).__init__()
        self.conv1 = nn.Conv1d( in_channels=993, out_channels=16,kernel_size=(5),stride=(1),padding=(2)) # Conv layer 
        self.pool1 = nn.MaxPool1d(kernel_size=(2)) # Pooling layer
        # self.dense = nn.Linear(384,1) # Dense Layer, flatten array
        self.dense = nn.Linear(192,1)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        # self.dropout = nn.Dropout(0.7)
        self.dropout = nn.Dropout(0.1)


    def forward(self, input_data):
        x = self.relu(self.conv1(input_data))
        x = self.pool1(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.dense(x)
        x = self.sigmoid(x)
        return x

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

def fit(trainloader, validloader, learning_rate, num_epochs,device, loss_fn):
    
        model = ShallowCNN().to(device)
        model = model.float()
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.916324)

        valid_loss_min = np.Inf 
        early_stopping_iter = 2000
        early_stopping_counter = 0


        for epoch in range(num_epochs):
            print(f'Starting epoch {epoch+1}')

            train_loss, train_acc = train_epoch(model, device, trainloader,loss_fn, optimizer)
            valid_loss, valid_acc = valid_epoch(model, device, validloader, loss_fn) 

            if  valid_loss < valid_loss_min:
                valid_loss_min = valid_loss
                best_model_state = deepcopy(model.state_dict())
                torch.save(best_model_state, f'./saved_models/cnn-optimised-exp7-{valid_acc:.2f}.pt')
            else:
                early_stopping_counter += 1

            if early_stopping_counter > early_stopping_iter:  # if not improving for n iterations then stop
                break

            print("Epoch:{}/{} Training Loss:{:.3f}  Validation Loss:{:.3f} Training Accuracy {:.2f}% Validation Accuracy {:.2f}%".format(epoch + 1, num_epochs, train_loss, valid_loss, \
                train_acc, valid_acc))
            
        return 

# %%
# SET MODEL HYPER-PARAMETERS
torch.manual_seed(42)
num_epochs = 300
learning_rate = 0.000100
batch_size = 160
loss_fn = nn.BCELoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_dataset = '.../train_ad.json'
trainloader, validloader = load_data(train_dataset, batch_size=batch_size) # get train_loader
fit(trainloader, validloader, learning_rate, num_epochs, device, loss_fn)

