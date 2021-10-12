# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

# %%
def dataloder(dataset):
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_subsampler)
    return trainloader, testloader


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

# %%
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

# %%
def train_epoch(model, trainloader, loss_fn, optimizer, device):
    train_loss, train_acc = 0.0, 0
    model.train()
    for epoch in range(0, num_epochs):
      print(f'Starting epoch {epoch+1}')
      for i, data in enumerate(trainloader, 0):
        inputs, targets = data
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
        return train_loss/len(trainloader), train_acc/len(trainloader)
            
def valid_epoch(model, testloader, loss_fn, device):
    valid_loss, valid_acc = 0.0, 0
    model.eval()
    for i, data in enumerate(testloader, 0):
        inputs, targets = data
        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.float)
        targets = targets.unsqueeze(1)
        prediction = model(inputs)
        loss = loss_fn(prediction,targets)
        accuracy =  binary_accuracy(prediction, targets)
        valid_loss += loss.item()
        valid_acc += accuracy.item()
        return valid_loss/len(testloader), valid_acc/len(testloader)




# %%
if __name__ == '__main__':
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
    kfold = KFold(n_splits=k_folds, shuffle=True)
    model = biLSTM(input_size, hidden_size, num_layers, label_size, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Start print
    print('--------------------------------')

# %%


def fit(model, train_loader, val_loader, loss_fn, optimiser, device, epochs):
    valid_loss_min = np.Inf
    for epoch in range (epochs):
        train_loss, train_acc = train(model, train_loader, loss_fn, optimiser, device) #train model
        valid_loss, valid_acc = evaluate(model, val_loader, loss_fn, device) # evaluate on validation set
        if valid_loss < valid_loss_min:
            valid_loss_min = valid_loss
            torch.save(model.state_dict(), f"{path_root}/newCNNlld.pth")
        print(f'Epoch {epoch:03}: | Train Loss: {train_loss:.3f} | Val Loss: {valid_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {valid_acc*100:.2f}%')
    return