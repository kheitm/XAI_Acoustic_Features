
# %%
# imported packages
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
from sklearn.metrics import confusion_matrix, classification_report



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


def load_data(filepath, batch_size):
    testdata = AD_Dataset(filepath)
    testloader = torch.utils.data.DataLoader(testdata, batch_size = batch_size, shuffle = True) 
    return testloader

# %%
# called model Class and associated functions

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
        self.dense = nn.Linear(192,1)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
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

# called evaluate function
def evaluate(model, testloader, device):
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.float)
            targets = targets.unsqueeze(1).cpu().numpy() # change correct class to numpy array on cpu
            prediction = model(inputs)
            prediction = torch.round(prediction)
            y_pred_list.append(prediction.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    y_pred_list = [item for sublist in y_pred_list for item in sublist]
    y_target_list = targets.tolist()
    y_target_list = [item for sublist in y_target_list for item in sublist]

    return  y_target_list, y_pred_list


# %%
# FIT MODEL
path_root = f"{os.getcwd()}"
batch_size = 160
filepath = '/mount/arbeitsdaten/thesis-dp-1/heitmekn/working/test_ad.json'
testloader = load_data(filepath, batch_size=batch_size)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ShallowCNN().to(device)

# %%
# load model
model_path = f"{path_root}/best_models/cnn-optimised-exp7-74.27.pt"
base = os.path.basename(model_path)
os.path.splitext(base)
model_name = os.path.splitext(base)[0]
model.load_state_dict(torch.load(model_path))
y_target_list, y_pred_list = evaluate(model, testloader, device)


# %%
#test model

print(model_name)
print(confusion_matrix(y_target_list, y_pred_list))
print(classification_report(y_target_list, y_pred_list))

# %%
# Save metrics 
df = pd.read_csv('/mount/arbeitsdaten/thesis-dp-1/heitmekn/working/params.csv')
name = 'cnn-optimised-exp7-74.27'
filtered = df[df['Model_Name'] == name].values.tolist()
lst = filtered[0][1:]

original_stdout = sys.stdout 
with open(f'{path_root}/metrics/{model_name}.txt', 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    print("Model name: ", model_name)
    print(f"Optuna Value: {lst[0]}")
    print(f"Batch Size: {lst[1]}")
    print(f"Dropout: {lst[2]}")
    print(f"Learning Rate: {lst[3]}")
    print(f"Momentum: {lst[4]}")
    print(f"Optimiser: {lst[5]}")
    print(os.linesep)
    print("Confusion Matrix: ")
    print(confusion_matrix(y_target_list, y_pred_list))
    print(os.linesep)
    print("Classification Report: ")
    print(classification_report(y_target_list, y_pred_list))
    sys.stdout = original_stdout

