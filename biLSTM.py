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
num_classes = 2
learning_rate = 0.001
batch_size = 16
num_epochs = 2
n_splits = 5 # Number of K-fold Splits
SEED = 42

# %%
# Create a bidirectional LSTM
class biLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(biLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # multiply by 2 to extend tensor to include info from backward and forward direction
        # define initial hidden state
        # x.size.(o) is the batch size/number of examples
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        # define initial cell state
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        # second output is a (hidden state, cell state)tuple which we do not use
        out, _ = self.lstm(x, (h0, c0))
        # this takes the last hidden state to send to the linear layer
        out = self.fc(out[:, -1, :])

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