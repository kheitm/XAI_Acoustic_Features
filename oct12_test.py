# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

# %%
def dataloader(dataset):
    for fold, (train_ids, valid_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_ids)
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        validloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_subsampler)
    return trainloader, validloader

# %%
class biLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, label_size, dropout):
        super(biLSTM, self).__init__()
        # self.hidden_size = hidden_size
        # self.num_layers = num_layers  #stacked LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # multiply by 2 to extend tensor to include info from backward and forward direction
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, label_size)
        self.act = nn.Sigmoid() 

        
    def forward(self, x):
        # ht = hidden state, ct = cell state
       out, (ht, ct) = self.lstm(x) 

       # OPTION 1
       out = self.fc1(out[:, -1, :])
       out = self.relu(out)
       out = self.droput(out)

       # OPTION 2
       # cat1 = torch.cat((out[:, -1, :self.hidden_size].squeeze(1), out[:, 0, self.hidden_size:].squeeze(1)), dim=1)
       # out = self.fc1(cat1)
       # out = self.relu(out)
       # out = self.droput(out)

       # OPTION 3
       # cat2 = torch.cat((ht[-2, :, :], ht[-1, :, :]), dim=1)
       # out = fc1(cat2)
       # out = self.relu(out)
       # out = self.droput(out)

       # THEN
       out = self.fc2(out)
       out = self.relu(out)
       out = self.droput(out)
       # out = self.act(out)
       return out


def reset_weights(m):
  # reset model weights to avoid weight leakage
  for layer in m.children():
    if hasattr(layer, 'reset_parameters'):
        print(f'Reset trainable parameters of layer = {layer}')
        layer.reset_parameters()


def binary_accuracy(prediction, target):
    preds = torch.round(prediction) # round to the nearest integer
    correct = (preds == target).float()
    accuracy = correct.sum()/len(correct)
    return accuracy


def train_epoch(model, trainloader, loss_fn, optimizer, device):
    train_loss, train_acc = 0.0, 0
    model.train()
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.float)
        targets = targets.unsqueeze(1)
        optimizer.zero_grad()
        prediction = model(inputs)
        loss = loss_fn(prediction, targets)
        accuracy = binary_accuracy(prediction, targets)
        loss.backward()
        optimizer.step()
        train_loss += train_loss.item()
        train_acc += accuracy.item()
        # agg_train_loss = train_loss/len(trainloader)
        # agg_train_acc = train_acc/len(trainloader)
        return train_loss, train_acc
            
def valid_epoch(model, validloader, loss_fn, device):
    valid_loss, valid_acc = 0.0, 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in validloader:
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.float)
            targets = targets.unsqueeze(1)
            prediction = model(inputs)
            loss = loss_fn(prediction,targets)
            accuracy =  binary_accuracy(prediction, targets)
            valid_loss += loss.item()
            valid_acc += accuracy.item()
            # agg_valid_loss = valid_loss/len(validloader)
            # agg_valid_acc = valid_acc/len(validloader)
            return valid_loss, valid_acc


# %%
dataset = '/Users/kathy-ann/thesis_old/lld_feats.json' 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
k_folds = 5
num_epochs = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 25 # #number of features in input
num_layers = 2
hidden_size = 25 #number of features in hidden state
label_size = 1
learning_rate = 0.001
batch_size = 10
dropout = 0.5
loss_fn = nn.BCEWithLogitsLoss()
results = {} # For fold results
torch.manual_seed(42)
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
kfold_dict = {}
model = biLSTM(input_size, hidden_size, num_layers, label_size, dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for fold, (train_ids, valid_ids) in enumerate(kfold.split(dataset)):
    print(f'FOLD {fold}')
    print('--------------------------------')
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_ids)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
    validloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_subsampler)

    model.apply(reset_weights)
    history = {'train_loss': [], 'valid_loss': [],'train_acc':[],'valid_acc':[]}

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, device, trainloader,loss_fn, optimizer)
        valid_loss, valid_acc = valid_epoch(model, device, validloader, loss_fn) 

        epoch_train_loss = train_loss / len(trainloader.sampler)
        epoch_train_acc = train_acc / len(trainloader.sampler) * 100
        epoch_valid_loss = valid_loss / len(validloader.sampler)
        epoch_valid_acc = valid_acc / len(validloader.sampler) * 100

        print("Epoch:{}/{} Training Loss:{:.3f}  Validation Loss:{:.3f} Training Accuracy {:.2f} \
            %  Validation Accuracy {:.2f} %".format(epoch + 1, num_epochs, epoch_train_loss, epoch_valid_loss, \
            epoch_train_acc, epoch_valid_acc))
        
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        history['train_acc'].append(train_acc)
        history['valid_acc'].append(valid_acc)
        kfold_dict['fold{}'.format(fold+1)] = history 

        save_path = f'./model-fold-{fold}.pth'
        torch.save(model.state_dict(), save_path)

torch.save(model,'k_cross_CNN.pt')  


                                                                                                             
                                                                                                             
                                                                                                             
                                                                                                             
                                                                                                             

