
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
# 1. Loaded data
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
filepath = f'{path_root}/test_ad.json'
testdata = AD_Dataset(filepath)
batch_size = testdata.len
testloader = torch.utils.data.DataLoader(testdata, batch_size=batch_size) 

# %%
# 2. Called model Class and associated functions

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
# 3. Loaded model
# Process data on cpu to avoid collisions
device = torch.device("cpu")
model = ShallowCNN()
model_path = f"{path_root}/best_models/cnn-optimised-exp7-74.27.pt"
base = os.path.basename(model_path)
os.path.splitext(base)
model_name = os.path.splitext(base)[0]
model.load_state_dict(torch.load(model_path, map_location=device))
y_target_list, y_pred_list = evaluate(model, testloader, device)

# Create indices for TPs and suubset TP test data
unq = np.array([x + 2*y for x, y in zip(y_pred_list, y_target_list)])
tp_idx = np.array(np.where(unq == 3)).tolist()[0]
shap_subset = Subset(testdata, tp_idx)
shaploader = DataLoader(shap_subset, batch_size=len(tp_idx)) #  batch_size=tp_idx.size
batch = next(iter(shaploader))
inputs, _ = batch
inputs = inputs.to(device, dtype=torch.float)
background = inputs[:20]
test_samples = inputs[20:35]

# Create DeepExplainer and extract shap values
# shape of values and test-samples must be the same
# Resample to timesteps x features (933 x 25)
e = shap.DeepExplainer(model=model, data=background)
shap_values = e.shap_values(test_samples) 
shap_values = shap_values[0].reshape(-1, 25)
test_samples_arr = np.array(test_samples)
test_samples_arr = test_samples_arr[0].reshape(-1, 25)

# produce Shap summary plot and save
# names = []
# shap.summary_plot(shap_values,test_samples_arr, feature_names=names, max_display=25)
plt.figure()
shap.summary_plot(shap_values,test_samples_arr, show=False, max_display=25)
plt.savefig(f'{path_root}/figures/{model_name}_shap_summary_plot.png')
plt.close()

# %%
# produce Shap bar plot and save
# names = []
plt.figure()
shap.summary_plot(shap_values, test_samples_arr, show=False, plot_type="bar", max_display=25 )
plt.savefig(f'{path_root}/figures/{model_name}_shap_bar_plot.png')
plt.close()
# %%
# Produce partial dependence plot
# Shows marginal effect one or two features have on the predicted outcome
# https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d
# https://www.kaggle.com/dansbecker/advanced-uses-of-shap-values
# https://towardsdatascience.com/explain-any-models-with-the-shap-values-use-the-kernelexplainer-79de9464897a
shap.dependence_plot("Feature 2", shap_values, test_samples_arr)
# %%
# Individual Value plot : Local interpretability
import pandas as pd
shap.initjs()

test_samples_arr = pd.DataFrame(test_samples_arr)
shap.force_plot(e.expected_value, shap_values[10], test_samples_arr .iloc[10,:])


# %%
plt.figure()
test_samples_arr = pd.DataFrame(test_samples_arr)
shap.force_plot(e.expected_value, shap_values[10], test_samples_arr .iloc[10,:], matplotlib=True, show=False)
plt.savefig(f'{path_root}/shap_force_plot.svg')
plt.close()
# %%
# You can also make a summary plot with only one feature for a quick result: shap.summary_plot(shap_values[0:1,:], X.iloc[0:1,:], color_bar=False)

# https://medium.com/dataman-in-ai/the-shap-with-more-elegant-charts-bc3e73fa1c0c
