import torch.nn as nn

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

    def forward(self, x):
        x = self.layer(x)

        return x