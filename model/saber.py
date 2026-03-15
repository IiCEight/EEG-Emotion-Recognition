import math

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.seed import SEED_RGNN_ADJACENCY_MATRIX
from model.grad_reverse import GradientReverse
from model.simple_graph_conv import SimpleGraphConv


class Saber(nn.Module):
    def __init__(self, num_electrodes=62, in_features=5, num_classes=3, num_layers=2, num_hidden=128,
                 dropout=0.7, domain_adaptation=False):
        super().__init__()

        self.num_electrodes = num_electrodes
        self.in_features = in_features
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.domain_adaptation = domain_adaptation
        self.num_classes = num_classes
        self.conv_out_dim = 32

        # used for weight to balance the contribution of the common and individual adjacency matrices
        self.omega = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.conv_l_1 = nn.Conv1d(self.in_features, self.conv_out_dim, kernel_size=1)
        self.conv_r_1 = nn.Conv1d(self.in_features, self.conv_out_dim, kernel_size=1)
        self.conv_l_2 = nn.Conv1d(self.in_features, self.conv_out_dim, kernel_size=1)
        self.conv_r_2 = nn.Conv1d(self.in_features, self.conv_out_dim, kernel_size=1)

        self.tanh = nn.Tanh()

        self.sgc = SimpleGraphConv(in_feature=self.in_features, out_feature=self.num_hidden, num_layers=self.num_layers)
        self.xs, self.ys = torch.tril_indices(self.num_electrodes, self.num_electrodes, offset=0)
        self.grad_reverse = GradientReverse()

        edge_weight = torch.tensor(SEED_RGNN_ADJACENCY_MATRIX).float()
        
        self.edge_weight = nn.Parameter(edge_weight[self.xs, self.ys], requires_grad=True)
        self.fc = nn.Linear(self.num_electrodes * self.num_hidden, self.num_hidden)
        self.fc2 = nn.Linear(self.num_hidden, self.num_classes)
        self.pool = global_add_pool
        self.init_weight()
        if self.domain_adaptation:
            self.domain_classifier = nn.Linear(self.num_hidden, 2)

    def init_weight(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x, alpha=0):
        adj_common = torch.zeros((self.num_electrodes, self.num_electrodes), device=x.device)
        adj_common[self.xs.to(adj_common.device), self.ys.to(adj_common.device)] = self.edge_weight
        adj_common = adj_common + adj_common.transpose(1, 0) - torch.diag(
            adj_common.diagonal())  # copy values from lower tri to upper tri
        # (batch, num_features, num_electrodes)
        x_T = rearrange(x, 'batch electrode features -> batch features electrode')

        node_l_1 = self.conv_l_1(x_T)
        node_r_1 = self.conv_r_1(x_T)
        node_l_2 = self.conv_l_2(x_T)
        node_r_2 = self.conv_r_2(x_T)

        adj_individual_1 = self.tanh(node_l_1.transpose(1, 2) @ node_r_1 / math.sqrt(self.conv_out_dim))
        adj_individual_2 = self.tanh(node_l_2.transpose(1, 2) @ node_r_2 / math.sqrt(self.conv_out_dim))
        adj_hybrid = adj_common + self.omega * ((adj_individual_1 + adj_individual_2) / 2)

        x = F.relu(self.sgc(x, adj_hybrid))

        # domain classification
        domain_output = None
        if self.domain_adaptation:
            reverse_x = self.grad_reverse(x, alpha)
            domain_output = self.domain_classifier(reverse_x)
        x = self.pool(x)
        # x = x.view(x.shape[0], -1)
        # x = F.dropout(x, p=self.dropout)
        # x = self.fc(x)
        x = F.dropout(x, p=self.dropout)
        x = self.fc2(x)
        return x, domain_output


def global_add_pool(x):
    """
    summing the output of each channel
    :param x: input x, shape of (batch, num_ele, num_hidden)
    :return: the result returned after the global and pool operation, shape of (batch, num_hidden)
    """
    return torch.sum(x, 1)