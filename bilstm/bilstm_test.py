
# %%
# 0. imported packages
import torch
import torch.nn as nn
import json
import os
import sys
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np

# %%
# 1. loaded test dataset
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


def load_data(filepath, batch_size):
    testdata = AD_Dataset(filepath)
    testloader = torch.utils.data.DataLoader(testdata, batch_size = batch_size, shuffle = True) 
    return testloader

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
        self.lstm = nn.LSTM(25, 25, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.attn = nn.TransformerEncoderLayer(d_model=25, nhead=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(25, 25)
        self.fc2 = nn.Linear(25, 2) # changed output to two nodes

        
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
# 3. FIT MODEL
path_root = f"{os.getcwd()}"
batch_size = 160
filepath = f"{path_root}/test_ad.json"
testloader = load_data(filepath, batch_size=batch_size)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = biLSTM().to(device)

# %%
# 4. Evaluate on Test data and create metrics/plots
model_directory = r'/mount/arbeitsdaten/thesis-dp-1/heitmekn/working/testmodels'
for entry in os.scandir(model_directory):
    model_path = entry.path
    base = os.path.basename(model_path)
    os.path.splitext(base)
    model_name = os.path.splitext(base)[0]
    model.load_state_dict(torch.load(model_path))
    y_target_list, y_pred_list = evaluate(model, testloader, device)
    cfm = confusion_matrix(y_target_list, y_pred_list)
    df = pd.read_csv(f"{path_root}/params.csv")
    filtered = df[df['Model_Name'] == model_name].values.tolist()
    lst = filtered[0][1:]

    # print metrics
    original_stdout = sys.stdout 
    with open(f'{path_root}/metrics/{model_name}.txt', 'w') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print("Model name: ", model_name)
        print(f"Batch Size: {lst[0]}")
        print(f"Dropout: {lst[1]}")
        print(f"Learning Rate: {lst[2]}")
        print(f"Momentum: {lst[3]}")
        print(f"Optimiser: {lst[4]}")
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
        ax.figure.savefig(f'{path_root}/figures/{model_name}_cfm.png')


# %%
