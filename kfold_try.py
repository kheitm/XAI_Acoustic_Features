# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold


# %%
# Set device and hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 25 # #number of features in input
num_layers = 2
hidden_size = 25 #number of features in hidden state
label_size = 1
learning_rate = 0.001
batch_size = 16
num_epochs = 2
dropout = 0.5
n_splits = 5 # Number of K-fold Splits
SEED = 42

# %%
# Create a bidirectional LSTM
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
# K-fold Cross Validation
def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()


# %%
if __name__ == '__main__':
  # Data
  dataset = '/Users/kathy-ann/thesis_old/lld_feats.json' 
  
  # Configuration options
  k_folds = 5
  num_epochs = 1
  loss_function = nn.BCEWithLogitsLoss()
  
  # For fold results
  results = {}
  
  # Set fixed random number seed
  torch.manual_seed(42)

  # Define the K-fold Cross Validator
  kfold = KFold(n_splits=k_folds, shuffle=True)

  # Start print
  print('--------------------------------')

  # %%

  # K-fold Cross Validation model evaluation
for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    
    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')
    
    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    
    # Define data loaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, sampler=train_subsampler)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=10, sampler=test_subsampler)
  
    # Init the neural network
    model = biLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Run the training loop for defined number of epochs
    for epoch in range(0, num_epochs):
      print(f'Starting epoch {epoch+1}')
      current_loss = 0.0
      # Iterate over the DataLoader for training data
      for i, data in enumerate(trainloader, 0):
        inputs, targets = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Print statistics
        current_loss += loss.item()
        if i % 500 == 499:
            print('Loss after mini-batch %5d: %.3f' % (i + 1, current_loss / 500))
            current_loss = 0.0
            
    # Process is complete.
    print('Training process has finished. Saving trained model.')


# %%
# Initialize network
model = biLSTM(input_size, hidden_size, num_layers, label_size, dropout).to(device)
criterion = nn.BCEWithLogitsLoss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# %%
def binary_accuracy(prediction, target):
    preds = torch.round(prediction) # round to the nearest integer
    correct = (preds == target).float()
    accuracy = correct.sum()/len(correct)
    return accuracy

def train(model, train_loader,loss_fn, optimiser, device):
    train_epoch_loss = 0
    train_epoch_acc = 0
    model.train()
    for input, target in train_loader:
        optimiser.zero_grad() # reset gradients for every batch
        input, target = input.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
        target = target.unsqueeze(1)
        prediction = model(input)
        train_loss = loss_fn(prediction, target)
        train_acc = binary_accuracy(prediction, target)
        train_loss.backward() # backpropogate
        optimiser.step() # update weights
        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()
    return train_epoch_loss/len(train_loader), train_epoch_acc/len(train_loader)


def evaluate(model, val_loader, loss_fn, device):
    val_epoch_loss = 0
    val_epoch_acc = 0
    model.eval() # deactivate dropout during evaluation
    with torch.no_grad(): #deactivate autograd
        for input, target in val_loader:
            input, target = input.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
            target = target.unsqueeze(1)
            prediction = model(input)
            val_loss = loss_fn(prediction, target)
            val_acc = binary_accuracy(prediction, target)
            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()
    return val_epoch_loss/len(val_loader), val_epoch_acc/len(val_loader)


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