import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, time, characteristics):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(time, time // 2, 3)
        self.conv2 = nn.Conv1d(time // 2, 1, 3)
        self.fc1 = nn.Linear(characteristics, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x) # reduce time dimensions
        x = F.relu(x) # apply relu

        x = self.conv2(x) # reduce time dimension to 1
        x = F.relu(x) # apply relu

        x = self.fc1(x) # project learned dimensions
        x = torch.flatten(x)

        return self.sig(x)
