import torch.nn as nn


class g(nn.Module):
    def __init__(self, num_electrodes, num_features, num_classes, adj_matrix):
        super().__init__()
        self.fc1 = nn.Linear(num_electrodes, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out