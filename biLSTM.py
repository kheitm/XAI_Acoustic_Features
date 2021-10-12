# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


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
        self.act = nn.BCEWithLogitsLoss() 

        
    def forward(self, x):
        # ht = hidden state, ct = cell state
       out, (ht, ct) = self.lstm(x) 
       out = self.fc1(out[:, -1, :])
       out = self.relu(out)
       out = self.droput(out)

       # OPTION 2
       # out = torch.cat((out[:, -1, :self.hidden_size].squeeze(1), out[:, 0, self.hidden_size:].squeeze(1)), dim=1)
       # out = self.fc1(out)

       # OPTION 3
       # ht = torch.cat((ht[-2, :, :], ht[-1, :, :]), dim=1)
       # out = fc1(ht)

       # THEN
       out = self.fc2(out)
       out = self.relu(out)
       out = self.droput(out)
       return out

# %%
# Load Data


# %%
# Initialize network
model = biLSTM(input_size, hidden_size, num_layers, num_classes).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# %%
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy  \
              {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()

# %%
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)