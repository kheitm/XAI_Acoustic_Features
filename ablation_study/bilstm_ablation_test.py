
# %%
# 0. imported packages
import torch
import torch.nn as nn
import json
import os
import sys
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

# %%
# 0. Set arguments

parser = argparse.ArgumentParser()
parser.add_argument("--feats",
                    type=int,
                    help="Feature left out of training data.")       # args.feats 
args = parser.parse_args()

# %%
# 1. loaded test dataset
class AD_Dataset():
    def __init__(self, FilePath): 
        self.FilePath = FilePath 
        
        with open(FilePath) as fileinput: 
            XY = json.load(fileinput) 
        X = np.array([Row[1] for Row in XY])
        X = np.delete(X, args.feats, axis=2)
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


def load_data(filepath, batch_size):
    testdata = AD_Dataset(filepath)
    testloader = torch.utils.data.DataLoader(testdata, batch_size = batch_size, shuffle = True) 
    return testloader

path_root = f"{os.getcwd()}"
batch_size = 146
filepath = f"{path_root}/test_ad.json"
testloader = load_data(filepath, batch_size=batch_size)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
# 2. Called model Class and associated functions

def binary_accuracy_softmax(prediction, target):
    preds = torch.argmax(torch.softmax(prediction, dim=1), dim=1)
    correct = (preds == target).float()
    accuracy = correct.sum()/len(correct)
    return accuracy


class biLSTM(nn.Module):
    def __init__(self):
        super(biLSTM, self).__init__()
        self.lstm = nn.LSTM(24, 24, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.attn = nn.TransformerEncoderLayer(d_model=24, nhead=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(24, 24)
        self.fc2 = nn.Linear(24, 2) 

        
    def forward(self, x):
        out,(hidden,_) = self.lstm(x)
        out = self.attn(hidden)
        out = out[-1, :, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def evaluate(model, testloader, device):
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.long) # change correct class to numpy array on cpu
            prediction = model(inputs)
            targets = targets.unsqueeze(1).cpu().numpy()
            preds = torch.argmax(torch.softmax(prediction, dim=1), dim=1)
            y_pred_list.append(preds.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    y_pred_list = [item for sublist in y_pred_list for item in sublist]
    y_target_list = targets.tolist()
    y_target_list = [item for sublist in y_target_list for item in sublist]

    return  y_target_list, y_pred_list

# %%
# 3. Generate metrics and plots

model = biLSTM().to(device)
model_path = f"{path_root}/saved_trials/bilstm_feat_#22_exp1-70.69.pt"
base = os.path.basename(model_path)
os.path.splitext(base)
model_name = os.path.splitext(base)[0]
model.load_state_dict(torch.load(model_path))

# generate predictions and target lists
y_target_list, y_pred_list = evaluate(model, testloader, device)
cfm = confusion_matrix(y_target_list, y_pred_list)

# print metrics
original_stdout = sys.stdout 
with open(f'{path_root}/best_bilstm_ablation_models/metrics/bilstm_feat_#22_{model_name}.txt', 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    print("Model name: ", model_name)
    print(f"Missing feature: {args.feats + 1}")
    print(os.linesep)
    print("Confusion Matrix: ")
    print(cfm)
    print(os.linesep)
    print('F1 score: %f' % f1_score(y_target_list, y_pred_list, average='micro'))
    print('Accuracy score: %f' % accuracy_score(y_target_list, y_pred_list))
    print("Classification Report: ")
    print(classification_report(y_target_list, y_pred_list))
    sys.stdout = original_stdout

    #  Save plots to file
    classes = ["No-AD", "AD"]
    plt.figure(figsize = (10,7))
    ax = sn.heatmap(cfm/np.sum(cfm), annot=True, fmt = '.2%',cmap='YlGnBu', annot_kws = {"size": 16})
    ax.xaxis.tick_top() # x axis on top
    ax.xaxis.set_label_position('top')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    ax.figure.savefig(f'{path_root}/best_bilstm_ablation_models/figures/{model_name}22_cfm.png')


