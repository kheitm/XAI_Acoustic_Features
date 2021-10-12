# %%
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
# import spacy
# import jovian
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import string
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import mean_squared_error

# %%
#input
x = torch.tensor([[1,2, 12,34, 56,78, 90,80],
                 [12,45, 99,67, 6,23, 77,82],
                 [3,24, 6,99, 12,56, 21,22]]) 

x.shape 
# %%
# Embedding layer: takes the vocabulary size and desired word-vector 
# length as input. You can optionally provide a padding index, 
# to indicate the index of the padding element in the 
# embedding matrix.
model1 = nn.Embedding(100, 7, padding_idx=0)
out1 = model1(x)
print(out1.shape)
print(out1)


# %%
# nn.LSTM
# pass the embedding layer’s output into an LSTM layer 
# (created using nn.LSTM), which takes as input the word-vector 
# length, length of the hidden state vector and number of layers. 
# Additionally, if the first element in our input’s shape 
# has the batch size, we can specify batch_first = True
model2 = nn.LSTM(input_size=7, hidden_size=3, num_layers=1, batch_first=True)

# %%

# %%
