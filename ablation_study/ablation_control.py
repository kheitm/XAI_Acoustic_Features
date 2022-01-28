
# %%
import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import argparse
import copy
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
from copy import deepcopy

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--ranseed",
                    type=int,
                    help="Random seed uused in training data.")       # args.feats 
parser.add_argument("--exp",
                    type=str,
                    help="Experiment number.")
args = parser.parse_args()
# %%
# Load dataset
class AD_Dataset():
    def __init__(self, FilePath): 
        self.FilePath = FilePath 
        
        with open(FilePath) as fileinput: 
            XY = json.load(fileinput) 
        X = np.array([Row[1] for Row in XY])
        # X = np.delete(X, args.feats, axis=2)
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
# Call model Class and associated functions
def load_data(filepath, batch_size):
    dataset = AD_Dataset(filepath)
    train_size = int(len(dataset) * 0.7)
    valid_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(args.ranseed))
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
    validloader = torch.utils.data.DataLoader(val_data, batch_size = batch_size, shuffle = True) #Passing the trainset into a Dataloader
    return trainloader, validloader


# Compute the accuracy based on two-node-output
def binary_accuracy_softmax(prediction, target):
    preds = torch.argmax(torch.softmax(prediction, dim=1), dim=1)
    correct = (preds == target).float()
    accuracy = correct.sum()/len(correct)
    return accuracy


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d( in_channels=993, out_channels=16,kernel_size=(5),stride=(1),padding=(2)) 
        self.pool1 = nn.MaxPool1d(kernel_size=(2)) 
        self.dense = nn.Linear(192,2)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.1)


    def forward(self, input_data):
        out = self.relu(self.conv1(input_data))
        out = self.pool1(out)
        out = self.dropout(out)
        out = torch.flatten(out, 1)
        out = self.dense(out)
        return out

# %%
def train_epoch(model,device, trainloader, loss_fn, optimizer):
    train_loss, train_acc = 0.0, 0
    model.train()
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.long)
        optimizer.zero_grad()
        prediction = model(inputs)
        loss = loss_fn(prediction, targets)
        accuracy = binary_accuracy_softmax(prediction, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        train_acc += accuracy.item()
    return train_loss/len(trainloader), train_acc/len(trainloader) * 100

def valid_epoch(model, device, validloader, loss_fn):
    valid_loss, valid_acc = 0.0, 0
    model.eval()
    with torch.no_grad():
      for inputs, targets in validloader:
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.long)
            prediction = model(inputs)
            loss = loss_fn(prediction, targets)
            accuracy =  binary_accuracy_softmax(prediction, targets) 
            valid_loss += loss.item() * inputs.size(0) # running loss
            valid_acc += accuracy.item()
      return valid_loss/len(validloader), valid_acc/len(validloader) * 100

# %%
# FIT MODEL

def fit(trainloader, validloader, num_epochs, device, loss_fn):
    
        model = CNN().to(device)

        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
        

        es_min_valid_loss = np.Inf 
        es_counter = 0
        es_patience = 10
        best_model_state = copy.deepcopy(model.state_dict())
        best_acc = 0.0


        for epoch in range(num_epochs):
            print('-' * 100)
            print(f'Starting epoch {epoch+1}')
            
            epoch_train_loss, epoch_train_acc = train_epoch(model, device, trainloader,loss_fn, optimizer)
            epoch_valid_loss, epoch_valid_acc = valid_epoch(model, device, validloader, loss_fn) 

            if epoch_valid_acc > best_acc:
                best_acc = epoch_valid_acc
                best_model_state = copy.deepcopy(model.state_dict())
                torch.save(best_model_state, f'./saved_trials/model_seed#{args.ranseed}_exp{args.exp}-{epoch_valid_acc:.2f}.pt')

            if  epoch_valid_loss < es_min_valid_loss:
                es_min_valid_loss = epoch_valid_loss
                es_counter = 0
            else:
                es_counter += 1
            

            if es_counter >= es_patience:  # if not improving for n iterations then stop
                print(f'Early stopping at epoch {epoch+1}!')
                break

            print("Epoch:{}/{} Training Loss {:.3f} Training Accuracy {:.3f} Validation Loss:{:.3f} Validation Accuracy {:.3f}".format(epoch + 1, num_epochs, epoch_train_loss, \
                epoch_train_acc, epoch_valid_loss, epoch_valid_acc))

        return 


# %%
# Initiate training
# torch.manual_seed(42)
num_epochs = 50
batch_size = 160
lr = 0.002374
loss_fn = nn.CrossEntropyLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_dataset = '/mount/arbeitsdaten/thesis-dp-1/heitmekn/working/train_ad.json'
trainloader, validloader = load_data(train_dataset, batch_size=batch_size) # get train_loader
fit(trainloader, validloader, num_epochs, device, loss_fn)

