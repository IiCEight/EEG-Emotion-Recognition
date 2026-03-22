

from torch import nn
from torch.nn import functional as F


class Classifier(nn.Module):
    def __init__(self, in_feature, num_class):
        super().__init__()
        self.classifier = nn.Linear(in_feature, num_class)

    def forward(self, x):
        x = self.classifier(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_feature, num_class=2):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(in_feature, in_feature)
        self.fc2 = nn.Linear(in_feature, num_class)
        self.dropout1 = nn.Dropout(p=0.25)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout1(x)
        x = self.fc2(x)
        # x = self.sigmoid(x)
        return x
