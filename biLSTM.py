# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


# %%
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%