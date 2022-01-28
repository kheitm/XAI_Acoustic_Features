
# %%
# imported packages
import torch
import torch.nn as nn
import numpy as np
import json
from torch.utils.data import DataLoader, Subset
import os
import matplotlib.pyplot as plt
import shap

# %%
# 1. Load data
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


path_root = f"{os.getcwd()}"
filepath_test = f'{path_root}/test_ad.json'
testdata = AD_Dataset(filepath_test)
filepath_train = f'{path_root}/train_ad.json'
traindata = AD_Dataset(filepath_train)
testloader = torch.utils.data.DataLoader(testdata, batch_size=testdata.len) 


# %%
# 2. Call model Class and associated functions

class biLSTM(nn.Module):
    def __init__(self):
        super(biLSTM, self).__init__()
        self.lstm = nn.LSTM(25, 25, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.attn = nn.TransformerEncoderLayer(d_model=25, nhead=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(25, 25)
        self.fc2 = nn.Linear(25, 2) 

        
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
# 3. Load model
# Process data on cpu to avoid collisions
device = torch.device("cpu")
model = biLSTM()
model_path = f"{path_root}/best_bilstm_models/bilstm-exp2-71.08.pt"
base = os.path.basename(model_path)
os.path.splitext(base)
model_name = os.path.splitext(base)[0]
model.load_state_dict(torch.load(model_path, map_location=device))


# %%
# Get TP, TN and TP + TN indices
y_target_list, y_pred_list = evaluate(model, testloader, device)
unq = np.array([x + 2*y for x, y in zip(y_pred_list, y_target_list)])
tp_idx = np.array(np.where(unq == 3)).tolist()[0]
tn_idx = np.array(np.where(unq == 0)).tolist()[0]
tptn_idx = np.concatenate((tp_idx, tn_idx))
# %%
# Get test data samples True Positives
shap_subset = Subset(testdata, tp_idx)
shaploader = DataLoader(shap_subset, batch_size=len(tp_idx)) #  batch_size=tp_idx.size
batch = next(iter(shaploader))
inputs, _ = batch
test_samples = inputs.to(device, dtype=torch.float)

# %%
# Get test data samples True Negatives
shap_subset = Subset(testdata, tn_idx)
shaploader = DataLoader(shap_subset, batch_size=len(tn_idx)) #  batch_size=tp_idx.size
batch = next(iter(shaploader))
inputs, _ = batch
test_samples = inputs.to(device, dtype=torch.float)

# %%
# Get test data samples TP + TN
shap_subset = Subset(testdata, tptn_idx)
shaploader = DataLoader(shap_subset, batch_size=len(tptn_idx)) #  batch_size=tp_idx.size
batch = next(iter(shaploader))
inputs, _ = batch
test_samples = inputs.to(device, dtype=torch.float)
# %%
# first 100 training examples as our background dataset to integrate over
trainloader = torch.utils.data.DataLoader(traindata, batch_size=100, shuffle=True) 
batch = next(iter(trainloader))
inputs, _ = batch
background = inputs.to(device, dtype=torch.float)

# %%
# Create DeepExplainer and extract shap values
# shape of values and test-samples must be the same
# Resample to timesteps x features (933 x 25)
e = shap.DeepExplainer(model=model, data=background)
shap_values = e.shap_values(test_samples)

# %%
# Reshape shap matrices to 2D
shap_values = np.array(shap_values[1])
shap_values.shape
shap_values = shap_values.reshape(-1, 25)
shap_values.shape
test_samples_arr = np.array(test_samples)
test_samples_arr = test_samples_arr.reshape(-1, 25)
test_samples_arr.shape

# %%
# produce Shap summary plot and save
names = ['Loudness', 'alphaRatio', 'hammarbergIndex', 'slope0-500','slope500-1500',\
        'spectralFlux', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4',\
        'F0semitoneFrom27.5Hz', 'Jitter_Local', 'Shimmer_Local', 'HNRdBACF', 'logRelF0-H1-H2',\
        'logRelF0-H1-A3', 'F1frequency', 'F1bandwidth', 'F1amplitudeLogRelF0_', 'F2frequency',\
        'F2bandwidth', 'F2amplitudeLogRelF0', 'F3frequency', 'F3bandwidth', 'F3amplitudeLogRelF0']
# shap.summary_plot(shap_values,test_samples_arr, feature_names=names, max_display=25)
plt.figure()
shap.summary_plot(shap_values,test_samples_arr, feature_names=names, show=False, max_display=25)
plt.savefig(f'{path_root}/thesis/bilstmTPTNsum.png', bbox_inches='tight')
plt.tight_layout()
plt.close()

# %%
# produce Shap bar plot and save
names = ['Loudness', 'alphaRatio', 'hammarbergIndex', 'slope0-500','slope500-1500',\
        'spectralFlux', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4',\
        'F0semitoneFrom27.5Hz', 'Jitter_Local', 'Shimmer_Local', 'HNRdBACF', 'logRelF0-H1-H2',\
        'logRelF0-H1-A3', 'F1frequency', 'F1bandwidth', 'F1amplitudeLogRelF0_', 'F2frequency',\
        'F2bandwidth', 'F2amplitudeLogRelF0', 'F3frequency', 'F3bandwidth', 'F3amplitudeLogRelF0']
plt.figure()
shap.summary_plot(shap_values, test_samples_arr, feature_names=names, show=False, plot_type="bar", max_display=25 )
plt.savefig(f'{path_root}/thesis/bilstmTPTNbar.png', bbox_inches='tight')
plt.close()


# Create dependency plot
# by passing show=False you can prevent shap.dependence_plot from calling
# the matplotlib show() function, and so you can keep customizing the plot
# before eventually calling show yourself
plt.figure()
names = ['Loudness', 'alphaRatio', 'hammarbergIndex', 'slope0-500','slope500-1500',\
        'spectralFlux', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4',\
        'F0semitoneFrom27.5Hz', 'Jitter_Local', 'Shimmer_Local', 'HNRdBACF', 'logRelF0-H1-H2',\
        'logRelF0-H1-A3', 'F1frequency', 'F1bandwidth', 'F1amplitudeLogRelF0_', 'F2frequency',\
        'F2bandwidth', 'F2amplitudeLogRelF0', 'F3frequency', 'F3bandwidth', 'F3amplitudeLogRelF0']

shap.dependence_plot(3, shap_values, test_samples_arr, interaction_index=3, show=False)
plt.xlabel("slope0-500")
plt.ylabel("SHAP value for the slope0-500")
plt.savefig(f'{path_root}/thesis/bilstmTPTNdep_slope0-500.png', bbox_inches='tight')
plt.show()
