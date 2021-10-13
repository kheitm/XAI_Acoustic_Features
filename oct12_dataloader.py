# %%
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from sklearn.model_selection import KFold
import numpy as np

# %%
dataset = '/Users/kathy-ann//thesis_old/subset20_lld_feats.json'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
loss_fn = nn.BCEWithLogitsLoss()
k_folds = 5
num_epochs = 10
input_size = 25 # #number of features in input
num_layers = 2
hidden_size = 25 #number of features in hidden state
label_size = 1
learning_rate = 0.001
batch_size = 10
dropout = 0.5
results = {} # For fold results
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
kfold_dict = {}

# %%
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
for fold, (train_ids, valid_ids) in enumerate(kfold.split(dataset)):
    print('------------fold no---------{}----------------------'.format(fold))
    train_subsampler = SubsetRandomSampler(train_ids)
    valid_subsampler = SubsetRandomSampler(valid_ids)
    trainloader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
    validloader = DataLoader(dataset, batch_size=batch_size, sampler=valid_subsampler)

# %%
print(validloader)
# %%
for i, data in enumerate(trainloader, 0):
    targets, inputs = data
    print(targets)

    
  

# %%
