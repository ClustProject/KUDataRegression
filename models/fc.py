import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from models.train_model import Train_Test


class FC(nn.Module):
    def __init__(self, representation_size, drop_out, bias):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(representation_size, 32, bias = bias)
        self.fc2 = nn.Linear(32, 1, bias = bias)
        self.layer = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(drop_out),
            self.fc2
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.layer(x)
        x = self.sigmoid(x)
        return x