# %%
""""
WORKING FILE: OCT 21

"""
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
import pandas as pd
import numpy as np
from numpy import array

# %%
# PREPARE TRAIN AND TEST DATA

class JSON_Dataset():
    def __init__(self, FilePath): 
        self.FilePath = FilePath 
        
        with open(FilePath) as fileinput: 
            XY = json.load(fileinput) 
        x = np.array([Row[1] for Row in XY])
        y = np.array([Row[0] for Row in XY])
        self.data = (y, x)
        self.len = len(self.data[0]) 
        self.shape = self.data[0].shape 
        
    def __getitem__(self, index):
        X = self.data[1][index] 
        Y = self.data[0][index] 
        return X, Y 

    def __len__(self):
        return self.len #Setting length attribute
    
    def __shape__(self): #Setting shape attribute
        return self.shape

filepath = '/mount/arbeitsdaten/thesis-dp-1/heitmekn/working/ad.json'
dataset = JSON_Dataset(filepath)
trainloader = DataLoader(dataset, batch_size=len(dataset))
train_np = np.array(trainloader.dataset, dtype=object)
train_np[:10]


# %%
# outline model

class biLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, label_size, dropout):
        super(biLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.3)
        self.attn = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, label_size)

        
    def forward(self, x):
        out,(hidden,_) = self.lstm(x)
        out = self.attn(hidden)
        out = out.mean(dim=0)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = biLSTM()

# %%
model.load_state_dict(torch.load('/content/drive/MyDrive/Colab/1_DL/best_models/classy2.pt'))