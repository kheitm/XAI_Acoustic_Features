
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
import matplotlib.pyplot as plt
import numpy as np

# %%
# 1. Load test dataset
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

path_root = f"{os.getcwd()}"
batch_size = 146
filepath = f"{path_root}/test_ad.json"
testloader = load_data(filepath, batch_size=batch_size)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
# 2. Call model Class and associated functions

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
        self.dropout = nn.Dropout()


    def forward(self, input_data):
        out = self.relu(self.conv1(input_data))
        out = self.pool1(out)
        out = self.dropout(out)
        out = torch.flatten(out, 1)
        out = self.dense(out)
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
model = CNN().to(device)
model_directory = r'/mount/arbeitsdaten/thesis-dp-1/heitmekn/working/saved_models'
for entry in os.scandir(model_directory):
    model_path = entry.path
    base = os.path.basename(model_path)
    os.path.splitext(base)
    model_name = os.path.splitext(base)[0]
    model.load_state_dict(torch.load(model_path))

    # generate predictions and target lists
    y_target_list, y_pred_list = evaluate(model, testloader, device)
    cfm = confusion_matrix(y_target_list, y_pred_list)

    # print metrics
    original_stdout = sys.stdout 
    with open(f'{path_root}/ablation_controls/metrics/{model_name}.txt', 'w') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print("Model name: ", model_name)
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
        ax.figure.savefig(f'{path_root}/ablation_controls/figures/{model_name}_cfm.png')


