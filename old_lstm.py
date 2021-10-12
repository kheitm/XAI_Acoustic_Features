# %%
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import string
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import mean_squared_error

# %%
#input
# tensor with random data and the supplied dimensionality 
# with torch.randn()
x = torch.randn((8, 933, 25))
print(x)

# %%
input_size = 25 # #number of features in input
num_layers = 2
hidden_size = 25 #number of features in hidden state
label_size = 1
learning_rate = 0.001
batch_size = 16
num_epochs = 2
n_splits = 5 # Number of K-fold Splits
SEED = 42
model = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

# %%
y = model(x)
# %%
print(type(y))
# %%
